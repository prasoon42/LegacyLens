#!/usr/bin/env python3
"""Direct test for generic 7-segment OCR"""
import sys
import cv2
from src.detect_and_ocr import detect_text

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

# Detect
items = detect_text(img)

print(f"\n=== DETECTED ITEMS ===")
for item in items:
    print(f"Text: '{item['text']}' (Conf: {item['conf']:.2f})")

if not items:
    print("No items detected.")
