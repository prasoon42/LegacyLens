#!/usr/bin/env python3
"""Direct test of OCR without Streamlit caching issues"""
import sys
import cv2
from src.detect_and_ocr import preprocess_image, detect_digits, map_rows_to_metrics

if len(sys.argv) < 2:
    print("Usage: python test_direct.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path)
if img is None:
    print(f"Could not read image: {img_path}")
    sys.exit(1)

print(f"Testing image: {img_path}")
print(f"Image shape: {img.shape}")

# Preprocess
img = preprocess_image(img)

# Detect
boxes = detect_digits(img)
print(f"Detected {len(boxes)} boxes")

# OCR
metrics, _ = map_rows_to_metrics(img, boxes)
print(f"\n=== FINAL RESULT ===")
print(f"SYS: {metrics.get('SYS', 'N/A')}")
print(f"DIA: {metrics.get('DIA', 'N/A')}")
print(f"PULSE: {metrics.get('PULSE', 'N/A')}")
