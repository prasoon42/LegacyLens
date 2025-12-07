# app/app.py
import sys
import pathlib
import time
import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
import cv2

# Add project root to path
project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Force reload to pick up code changes
import importlib
import src.detect_and_ocr
importlib.reload(src.detect_and_ocr)

from src.detect_and_ocr import preprocess_image, detect_digits, reconstruct_reading, visualize, apply_preprocessing, map_rows_to_metrics

st.set_page_config(page_title="BioNexus Reader", layout="wide", page_icon="🧬")

# Load Premium CSS
css_path = pathlib.Path(__file__).parent / "style_premium.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session State
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.title("🧬 BioNexus")
    st.caption("Advanced Medical OCR Engine")
    
    st.markdown("### ⚙️ Configuration")
    preprocess_mode = st.selectbox(
        "Enhancement Mode",
        ["Default", "High Contrast", "Thresholding", "Grayscale", "Denoise"],
        index=1, # Default to High Contrast as it works better
        help="Select an image enhancement algorithm to improve detection."
    )
    
    st.markdown("### 📐 Region of Interest")
    enable_crop = st.checkbox("Manual Crop", value=False)
    
    if enable_crop:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            crop_y = st.number_input("Top (%)", 0, 90, 20)
            crop_x = st.number_input("Left (%)", 0, 90, 20)
        with col_c2:
            crop_h = st.number_input("Height (%)", 10, 100, 60)
            crop_w = st.number_input("Width (%)", 10, 100, 60)
    else:
        crop_y, crop_x, crop_h, crop_w = 0, 0, 100, 100

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Main Layout
col_left, col_right = st.columns([1.2, 0.8], gap="large")

with col_left:
    st.markdown("## 📸 Input Feed")
    img_file = st.file_uploader("Upload Medical Display Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if img_file:
        pil_img = Image.open(img_file).convert("RGB")
        img = np.array(pil_img)
        
        # Crop Logic
        h, w = img.shape[:2]
        cy = int(h * (crop_y / 100))
        cx = int(w * (crop_x / 100))
        ch = int(h * (crop_h / 100))
        cw = int(w * (crop_w / 100))
        
        # Ensure valid crop
        if ch < 10: ch = 10
        if cw < 10: cw = 10
        
        img_cropped = img[cy:cy+ch, cx:cx+cw]
        
        # Display Image
        st.image(img_cropped, caption="Analysis Region", use_container_width=True)
        
        # Analyze Button
        if st.button("⚡ Analyze Reading", type="primary", use_container_width=True):
            with st.spinner("Running Neural Inference..."):
                # 1. Preprocess
                img_p = preprocess_image(img_cropped)
                img_processed = apply_preprocessing(img_p, mode=preprocess_mode)
                
                # 2. Detect
                boxes = detect_digits(img_processed)
                
                # 3. Reconstruct
                img_processed_for_detection = apply_preprocessing(img_p, mode=preprocess_mode)

                # 1. Detect Screen
                boxes = detect_digits(img_processed_for_detection)
                
                # 2. OCR with new Otsu logic
                # Now returns (readings, debug_img)
                # Use the original cropped image for OCR, as map_rows_to_metrics handles its own preprocessing
                readings, debug_img = map_rows_to_metrics(img_cropped, boxes)
                
                # Update History
                if readings:
                    new_entry = {
                        "Time": time.strftime("%H:%M"),
                        "SYS": readings.get('SYS', '--'),
                        "DIA": readings.get('DIA', '--'),
                        "PULSE": readings.get('PULSE', '--')
                    }
                    st.session_state.history.insert(0, new_entry)
                
                # Store results in session state to persist across reruns if needed
                st.session_state.last_result = {
                    "readings": readings,
                    "debug_img": debug_img,
                    "boxes": boxes,
                    "original_cropped_img": img_cropped # Store original for visualization
                }

with col_right:
    st.markdown("## 📊 Digital Readout")
    
    if 'last_result' in st.session_state:
        res = st.session_state.last_result
        readings = res['readings']
        debug_img = res['debug_img']
        boxes = res['boxes']
        original_cropped_img = res['original_cropped_img']

        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown("### 🔍 Detection Overlay")
            # Use the original cropped image for visualization
            annotated_img = visualize(original_cropped_img.copy(), boxes, str(readings))
            st.image(annotated_img, use_container_width=True)
            
            if debug_img is not None:
                st.markdown("### 🛠️ OCR Input (Debug)")
                st.image(debug_img, use_container_width=True, caption="What the AI sees (Otsu Threshold)")

        with col_res2:
            st.markdown("### 📊 Digital Readout")
            
            sys_val = readings.get('SYS', '--')
            dia_val = readings.get('DIA', '--')
            pulse_val = readings.get('PULSE', '--')
            
            st.markdown(f"""
            <div class="digital-display">
                <div class="metric-row">
                    <div class="metric-label">SYSTOLIC<br><span style="font-size:0.6em;opacity:0.7">mmHg</span></div>
                    <div class="metric-value">{sys_val}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">DIASTOLIC<br><span style="font-size:0.6em;opacity:0.7">mmHg</span></div>
                    <div class="metric-value">{dia_val}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">PULSE RATE<br><span style="font-size:0.6em;opacity:0.7">BPM</span></div>
                    <div class="metric-value" style="color:#ff0055">{pulse_val}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        # Placeholder State
        st.markdown("""
        <div class="digital-display" style="opacity: 0.5;">
            <div class="metric-row"><div class="metric-name">Systolic</div><div class="metric-value">--</div></div>
            <div class="metric-row"><div class="metric-name">Diastolic</div><div class="metric-value">--</div></div>
            <div class="metric-row"><div class="metric-name">Pulse</div><div class="metric-value">--</div></div>
        </div>
        <div style="margin-top: 1rem; color: #64748b; text-align: center;">
            Waiting for analysis...
        </div>
        """, unsafe_allow_html=True)

# History Table at Bottom
st.markdown("---")
st.markdown("### 📜 Recent Readings")
if st.session_state.history:
    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
        hide_index=True
    )
else:
    st.caption("No readings recorded yet.")

