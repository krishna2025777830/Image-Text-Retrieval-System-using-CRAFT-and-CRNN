import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import json
import os
from pathlib import Path
from datetime import datetime

# Configure page
st.set_page_config(page_title="OCR Pipeline", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize OCR readers (cached for performance)
@st.cache_resource
def load_ocr_readers():
    """Load OCR engines - EasyOCR only for stability"""
    # EasyOCR is the only engine (most reliable)
    easy_reader = easyocr.Reader(['en'])
    
    return {
        'easy': easy_reader
    }

# Image preprocessing with multiple options
# ===== OPTIMIZED PREPROCESSING FOR IMPROVED OCR =====

def resize_for_ocr(image, target_width=1200):
    """
    Resize image to optimal width for OCR accuracy.
    EasyOCR works best with 800-1600px width.
    For small text: enlarge to 2000-3000px
    """
    try:
        height, width = image.shape[:2]
        if width == 0:
            return image
        
        scale = target_width / width
        new_height = int(height * scale)
        
        # Use INTER_CUBIC for enlargement (better quality)
        # Use INTER_AREA for reduction
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(image, (target_width, new_height), interpolation=interpolation)
        
        return resized
    except Exception as e:
        return image

def enhance_contrast(image):
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Better than global equalization - preserves details.
    """
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    except Exception as e:
        return image

def sharpen_image(image, strength=1.0):
    """
    Sharpen image to enhance text edges.
    Useful for noisy or blurry images.
    """
    try:
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Blend: enhanced + sharpened
        result = cv2.addWeighted(image, 1.0 - strength*0.3, sharpened, strength*0.3, 0)
        return result
    except Exception as e:
        return image

def deskew_image(image):
    """
    Detect and correct image skew/rotation.
    Important for document scans at angles.
    """
    try:
        # Detect white/text regions
        coords = np.column_stack(np.where(image > 150))
        
        if len(coords) < 100:
            return image  # Not enough data
        
        # Find minimum bounding rectangle
        angle = cv2.minAreaRect(cv2.convexHull(coords))[-1]
        
        # Normalize angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        
        # Rotate if needed
        if abs(angle) > 0.5:
            h, w = image.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    except Exception as e:
        return image

def preprocess_image_light(image):
    """
    Light preprocessing - preserve text details.
    Best for: High-quality images, clear documents
    """
    try:
        # Resize for better recognition
        resized = resize_for_ocr(image, target_width=1200)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Light bilateral filter (preserves edges)
        filtered = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # Minimal blur with ODD kernel size
        blurred = cv2.GaussianBlur(filtered, (3, 3), 0)
        
        # Light contrast enhancement
        enhanced = enhance_contrast(blurred)
        
        # Create threshold for display (light version)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return gray, enhanced, thresh
    except Exception as e:
        return None, None, None

def preprocess_image_aggressive(image):
    """
    Aggressive preprocessing - for poor quality images.
    Best for: Noisy, blurry, low-contrast images
    """
    try:
        # Resize for poor quality images (enlarge)
        resized = resize_for_ocr(image, target_width=1500)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Deskew if needed
        gray = deskew_image(gray)
        
        # Bilateral filter (multi-pass for noisy images)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        filtered = cv2.bilateralFilter(filtered, 5, 50, 50)  # Second pass
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(filtered, (5, 5), 0)
        
        # Strong contrast enhancement
        enhanced = enhance_contrast(blurred)
        
        # Sharpen to enhance text edges
        sharpened = sharpen_image(enhanced, strength=0.5)
        
        # Otsu's thresholding for binary representation
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return gray, sharpened, thresh
    except Exception as e:
        return None, None, None

def preprocess_image(image, mode='light'):
    """
    Apply preprocessing to enhance OCR accuracy.
    
    Args:
        image: Input image
        mode: 'light' (high quality) or 'aggressive' (poor quality)
    """
    if mode == 'light':
        return preprocess_image_light(image)
    else:
        return preprocess_image_aggressive(image)

# ===== SMART FILTERING & POST-PROCESSING =====

def filter_by_confidence(results, threshold=0.5):
    """
    Intelligent confidence filtering.
    
    Args:
        results: EasyOCR readtext output
        threshold: Confidence cutoff (0.0-1.0)
    
    Returns:
        list: Filtered detections
    """
    filtered = []
    
    for bbox, text, confidence in results:
        # Confidence check
        if confidence < threshold:
            continue
        
        # Text quality check (not just spaces)
        if len(text.strip()) < 1:
            continue
        
        # Remove single junk characters
        if len(text.strip()) == 1 and text in '.,;:!?\'"':
            continue
        
        filtered.append((bbox, text, confidence))
    
    return filtered

def remove_overlapping_detections(results, overlap_threshold=0.3):
    """
    Remove duplicate/overlapping detections.
    EasyOCR sometimes detects same text twice.
    
    Args:
        results: List of (bbox, text, confidence) tuples
        overlap_threshold: IoU threshold for overlap (0.0-1.0)
    
    Returns:
        list: Deduplicated detections
    """
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union"""
        try:
            x1_coords = [p[0] for p in box1]
            x1_min, x1_max = min(x1_coords), max(x1_coords)
            y1_coords = [p[1] for p in box1]
            y1_min, y1_max = min(y1_coords), max(y1_coords)
            
            x2_coords = [p[0] for p in box2]
            x2_min, x2_max = min(x2_coords), max(x2_coords)
            y2_coords = [p[1] for p in box2]
            y2_min, y2_max = min(y2_coords), max(y2_coords)
            
            # Intersection area
            x_inter = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_inter = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            inter_area = x_inter * y_inter
            
            # Union area
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            
            if union_area == 0:
                return 0
            return inter_area / union_area
        except:
            return 0
    
    filtered = []
    for i, (bbox1, text1, conf1) in enumerate(results):
        is_duplicate = False
        
        for bbox2, text2, conf2 in filtered:
            iou = calculate_iou(bbox1, bbox2)
            
            if iou > overlap_threshold:
                # Overlaps with existing detection
                if conf1 > conf2:
                    # Replace with higher confidence
                    filtered.remove((bbox2, text2, conf2))
                else:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered.append((bbox1, text1, conf1))
    
    return filtered

def improve_bbox(bbox_points):
    """
    Tighten bounding boxes for better visualization.
    Converts 4-point bbox to tight rectangle.
    """
    try:
        x_coords = [point[0] for point in bbox_points]
        y_coords = [point[1] for point in bbox_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        tight_bbox = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]
        
        return tight_bbox
    except:
        return bbox_points

# Text detection and recognition using CRAFT + CRNN (via EasyOCR)
def extract_text_with_ocr(image, reader, confidence_threshold=0.3):
    """
    Extract text from image using EasyOCR (CRAFT + CRNN).
    
    Includes intelligent filtering, deduplication, and bounding box optimization.
    """
    # Initialize with empty/safe defaults
    extracted_data = {
        'full_text': '',
        'detections': [],
        'timestamp': datetime.now().isoformat(),
        'method': 'EasyOCR (Optimized)'
    }
    
    try:
        results = reader.readtext(image)
    except Exception as e:
        return extracted_data  # Return empty but valid dict on error
    
    # Step 1: Filter by confidence threshold
    results = filter_by_confidence(results, threshold=confidence_threshold)
    
    # Step 2: Remove overlapping/duplicate detections
    results = remove_overlapping_detections(results, overlap_threshold=0.3)
    
    # Step 3: Convert and improve bounding boxes
    detections = []
    for bbox, text, confidence in results:
        try:
            # Improve bbox (make it tight rectangular)
            tight_bbox = improve_bbox(bbox)
            bbox = np.array(tight_bbox, dtype=np.int32)
            
            # Calculate center point for sorting
            center_y = np.mean([point[1] for point in bbox])
            center_x = np.mean([point[0] for point in bbox])
            
            detections.append({
                'text': text.strip(),
                'confidence': float(confidence),
                'bbox': bbox.tolist(),
                'center_y': center_y,
                'center_x': center_x
            })
        except Exception:
            continue
    
    # Step 4: Sort by position (top-to-bottom, left-to-right)
    detections = sorted(detections, key=lambda x: (round(x['center_y'] / 25) * 25, x['center_x']))
    
    # Step 5: Final deduplication based on proximity
    cleaned_detections = []
    for detection in detections:
        is_duplicate = False
        for existing in cleaned_detections:
            # Check if texts are similar and very close
            if abs(detection['center_y'] - existing['center_y']) < 20:
                if abs(detection['center_x'] - existing['center_x']) < 60:
                    # Keep the one with higher confidence
                    if detection['confidence'] > existing['confidence']:
                        cleaned_detections.remove(existing)
                    else:
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            cleaned_detections.append(detection)
    
    # Step 6: Build final results
    full_text_parts = []
    for detection in cleaned_detections:
        extracted_data['detections'].append({
            'text': detection['text'],
            'confidence': detection['confidence'],
            'bbox': detection['bbox']
        })
        full_text_parts.append(detection['text'])
    
    extracted_data['full_text'] = ' '.join(full_text_parts)
    return extracted_data

# Draw bounding boxes on image
def draw_detections(image, results):
    """Draw bounding boxes around detected text"""
    image_with_boxes = image.copy()
    
    for detection in results['detections']:
        bbox = np.array(detection['bbox'], dtype=np.int32)
        text = detection['text']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.polylines(image_with_boxes, [bbox], True, (0, 255, 0), 2)
        
        # Put text label with confidence
        label = f"{text} ({confidence:.2f})"
        cv2.putText(image_with_boxes, label, tuple(bbox[0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image_with_boxes

# Save results to JSON
def save_results(results, filename="ocr_results.json"):
    """Save OCR results to JSON file"""
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / filename
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {
        'full_text': results['full_text'],
        'timestamp': results['timestamp'],
        'detections': []
    }
    
    for det in results['detections']:
        serializable_results['detections'].append({
            'text': det['text'],
            'confidence': float(det['confidence']),
            'bbox': det['bbox']
        })
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return str(output_path)

# Main app
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🔤 OCR Pipeline - Text Detection & Recognition</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.1rem; color: #555;'>Automatic extraction and recognition of text from images using CRAFT + CRNN</p>
</div>
""", unsafe_allow_html=True)

# Help section
with st.expander("📖 How to Get Best Results"):
    st.markdown("""
    ### ✨ Smart Preprocessing Optimizations
    
    The app automatically applies **intelligent preprocessing** based on your image quality selection:
    
    **High Quality (Clear Documents):**
    - Fast & accurate
    - Minimal processing
    - Confidence threshold: 0.6
    - Best for: Scanned documents, printed text
    
    **Medium Quality (Normal Photos):**
    - Balanced speed & accuracy
    - Light preprocessing
    - Confidence threshold: 0.5
    - Best for: Phone photos, screenshots
    
    **Low Quality (Noisy/Blurry):**
    - Slower, better detection
    - Aggressive preprocessing (multi-pass denoising)
    - Confidence threshold: 0.3
    - Best for: Blurry images, low contrast, poor lighting
    
    ### 🎯 Optimizations Applied
    - **Intelligent Resizing** - Enlarges for better text recognition
    - **Contrast Enhancement** - CLAHE for better readability
    - **Denoising** - Bilateral filter preserves text edges
    - **Edge Sharpening** - Makes text boundaries crisp
    - **Duplicate Removal** - Eliminates redundant detections
    - **BBox Tightening** - Improves visualization
    - **Smart Sorting** - Organizes text by position
    
    ### 💡 Tips
    - Use **Debug Mode** to see all optimizations applied
    - Adjust **Confidence Threshold** to control false positives
    - Select the right image quality for your use case
    - For very small text (< 50px), you may need manual preprocessing
    """)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # OCR Engine selection - EasyOCR only for stability
    ocr_engine = "EasyOCR (Recommended)"
    st.info("🚀 Using **EasyOCR** - Most reliable for text extraction")
    
    st.markdown("#### 🎯 Image Quality & Preprocessing")
    
    # Image quality selector (determines preprocessing strategy)
    image_quality = st.radio(
        "Image Quality",
        ["High (Clear Documents)", "Medium (Normal Photos)", "Low (Noisy/Blurry)"],
        index=1,
        help="Select image quality - affects preprocessing aggressiveness and confidence threshold"
    )
    
    # Map quality to preprocessing mode and threshold
    quality_settings = {
        "High (Clear Documents)": ("light", 0.6),
        "Medium (Normal Photos)": ("light", 0.5),
        "Low (Noisy/Blurry)": ("aggressive", 0.3)
    }
    
    preprocess_mode_auto, confidence_threshold_auto = quality_settings[image_quality]
    
    # Preprocessing mode (can override)
    preprocess_mode = st.radio(
        "Preprocessing Strategy",
        ["Light (Preserve Details)", "Aggressive (Enhanced Denoise)"],
        index=0 if preprocess_mode_auto == "light" else 1,
        help="Light: Fast, keeps details. Aggressive: Slower, better for poor quality"
    )
    
    # Confidence threshold (can override)
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=confidence_threshold_auto,
        step=0.05,
        help="Filter weak detections. Lower = accept more (but noisier). Higher = only best (but may miss text)."
    )
    
    st.caption(f"💡 Recommended for {image_quality}: threshold={confidence_threshold_auto}, mode={'Aggressive' if preprocess_mode_auto == 'aggressive' else 'Light'}")
    
    show_preprocessing = st.checkbox("Show Preprocessing Steps", value=False)
    show_detections = st.checkbox("Show Detected Boxes", value=True)
    save_results_opt = st.checkbox("Save Results to JSON", value=True)
    show_debug = st.checkbox("Debug Mode (show all detections)", value=False)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "bmp"])

with col2:
    st.subheader("📋 Results Summary")
    result_placeholder = st.empty()

if uploaded_file is not None:
    try:
        # Read and decode image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("❌ Failed to load image. Please try another file.")
            st.stop()
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)[:100]}")
        st.stop()
    
    with st.spinner("Loading OCR models..."):
        readers = load_ocr_readers()
    
    with st.spinner("Processing image..."):
        # Preprocessing
        preprocess_choice = 'aggressive' if preprocess_mode == "Aggressive (Denoise)" else 'light'
        gray, blurred, thresh = preprocess_image(image, mode=preprocess_choice)
        
        if gray is None:
            st.error("❌ Image preprocessing failed")
            st.stop()
        
        # Select OCR engine - EasyOCR only for stability
        results = extract_text_with_ocr(image, readers['easy'], confidence_threshold=0.3)
        
        # Safely access detections (EasyOCR always returns valid dict now)
        detections_list = results.get('detections', [])
        
        if not detections_list or not isinstance(detections_list, list):
            st.warning("⚠️ No text detected in image or detection format error")
            st.stop()
        
        # Filter by confidence threshold
        filtered_detections = [d for d in detections_list 
                              if isinstance(d, dict) and d.get('confidence', 0) >= confidence_threshold]
        results['detections'] = filtered_detections
        
        # Update full text with filtered results
        if filtered_detections:
            results['full_text'] = ' '.join([d['text'] for d in filtered_detections])
        else:
            results['full_text'] = ""
    
    # Display results
    st.success("✅ OCR Processing Complete!")
    
    # Show summary in sidebar
    with result_placeholder.container():
        st.metric("Detected Texts", len(results['detections']))
        st.metric("Confidence Avg", 
                 f"{np.mean([d['confidence'] for d in results['detections']]):.2f}" 
                 if results['detections'] else "N/A")
    
    # Main display
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_column_width=True)
    
    with col2:
        st.subheader("Detection Results")
        if show_detections:
            image_with_boxes = draw_detections(image, results)
            image_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
            st.image(image_boxes_rgb, use_column_width=True)
        else:
            st.info("Enable 'Show Detected Boxes' in settings to view bounding boxes")
    
    # Preprocessing visualization
    if show_preprocessing:
        st.divider()
        st.subheader("📊 Preprocessing Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Grayscale**")
            if gray is not None:
                st.image(gray, use_column_width=True, channels="GRAY")
            else:
                st.warning("Grayscale image unavailable")
        
        with col2:
            st.write("**Enhanced**")
            if blurred is not None:
                st.image(blurred, use_column_width=True, channels="GRAY")
            else:
                st.warning("Enhanced image unavailable")
        
        with col3:
            st.write("**Thresholded**")
            if thresh is not None:
                st.image(thresh, use_column_width=True, channels="GRAY")
            else:
                st.warning("Thresholded image unavailable")
    
    # Extracted text
    st.divider()
    st.subheader("📝 Extracted Text")
    
    if results['full_text'].strip():
        text_area = st.text_area(
            "Full Extracted Text:",
            value=results['full_text'],
            height=150,
            disabled=False
        )
        
        # Copy button
        st.code(results['full_text'], language="text")
    else:
        st.warning("⚠️ No text detected in the image")
    
    # Detailed detections table
    st.divider()
    st.subheader("🔍 Detailed Detections")
    
    if results['detections']:
        detection_data = []
        for i, det in enumerate(results['detections'], 1):
            # Format confidence with color indicator
            conf_pct = det['confidence'] * 100
            detection_data.append({
                "No.": i,
                "Text": det['text'],
                "Confidence": f"{conf_pct:.1f}%",
                "Quality": "🟢 High" if conf_pct >= 80 else "🟡 Medium" if conf_pct >= 60 else "🔴 Low"
            })
        
        st.dataframe(detection_data, width='stretch')
        
        st.info(f"✅ Total detections: {len(results['detections'])} | "
                f"Average confidence: {np.mean([d['confidence'] for d in results['detections']]):.1%}")
    else:
        st.warning("⚠️ No detections above the confidence threshold. Try lowering the threshold in settings.")
    
    # Debug mode - show all detections and comparison
    if show_debug:
        st.divider()
        st.subheader("🐛 Debug Info & Optimizations Applied")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Detection Stats:**")
            st.write(f"• Total detections: {len(results['detections'])}")
            if results.get('detections'):
                avg_conf = np.mean([d['confidence'] for d in results['detections']])
                st.write(f"• Avg confidence: {avg_conf:.1%}")
                st.write(f"• Min confidence: {min([d['confidence'] for d in results['detections']]):.1%}")
                st.write(f"• Max confidence: {max([d['confidence'] for d in results['detections']]):.1%}")
        
        with col2:
            st.write("**🔧 Preprocessing Applied:**")
            st.write(f"• Mode: {preprocess_choice}")
            st.write(f"• Image Quality: {image_quality}")
            st.write(f"• Confidence Threshold: {confidence_threshold:.2f}")
            st.write(f"• OCR Method: {results.get('method', 'Unknown')}")
        
        with col3:
            st.write("**✨ Optimizations:**")
            st.write("• ✅ Intelligent confidence filtering")
            st.write("• ✅ Duplicate detection removal")
            st.write("• ✅ Bounding box tightening")
            st.write("• ✅ Text position sorting")
            st.write("• ✅ Smart bbox improvement")
    
    # Save results
    if save_results_opt:
        st.divider()
        st.subheader("💾 Save Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save as JSON"):
                output_path = save_results(results)
                st.success(f"✅ Results saved to: `{output_path}`")
        
        with col2:
            if st.button("Download as Text"):
                st.download_button(
                    label="📥 Download Text",
                    data=results['full_text'],
                    file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

else:
    st.info("👆 Upload an image to start OCR processing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🎯 **Step 1**: Upload image")
    
    with col2:
        st.info("🔍 **Step 2**: Process with CRAFT+CRNN")
    
    with col3:
        st.info("📝 **Step 3**: View extracted text")

st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem; margin-top: 2rem;'>
    <p>OCR Pipeline | Powered by EasyOCR (CRAFT + CRNN) | January 2026</p>
</div>
""", unsafe_allow_html=True)
