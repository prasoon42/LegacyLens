import cv2
import numpy as np
import easyocr

def get_reader():
    return easyocr.Reader(['en'], gpu=False)

def detect_contours(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return

    # 1. Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to connect broken segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 2. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_boxes = []
    H, W = img.shape[:2]
    img_area = H * W
    
    min_area = img_area * 0.001 # 0.1% of image
    max_area = img_area * 0.1   # 10% of image (digits aren't huge)

    vis = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(h) / w
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
            
        # Filter by aspect ratio (digits are usually taller than wide, but 7-segment can be square-ish)
        # 1 (1:1) to 0.2 (5:1 height:width)
        if aspect_ratio < 0.8 or aspect_ratio > 5.0:
            continue

        # Draw candidate
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,255), 1)
        digit_boxes.append((x, y, x+w, y+h))

    print(f"Found {len(digit_boxes)} candidate contours")
    
    # 3. OCR on candidates
    reader = get_reader()
    final_readings = []
    
    # Sort boxes by Y then X to group them
    digit_boxes.sort(key=lambda b: (b[1] // 20, b[0])) # Group by row (approx 20px tolerance)

    for (x1, y1, x2, y2) in digit_boxes:
        # Add padding
        pad = 5
        crop = img[max(0, y1-pad):min(H, y2+pad), max(0, x1-pad):min(W, x2+pad)]
        
        try:
            res = reader.readtext(crop, detail=0, allowlist='0123456789.')
            text = "".join(res)
            if text.strip():
                print(f"Box {x1,y1,x2,y2} -> {text}")
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                final_readings.append(text)
        except Exception:
            pass

    cv2.imwrite("contour_debug.jpg", vis)
    print("Saved contour_debug.jpg")

if __name__ == "__main__":
    detect_contours("test_image.png")
