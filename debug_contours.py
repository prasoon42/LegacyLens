import cv2
import numpy as np
import sys

def debug_contours(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image")
        return

    # Run YOLO detection first (simulating detect_and_ocr.py)
    try:
        from ultralytics import YOLO
        import os
        weights_path = os.path.abspath("weights/yolov8n.pt")
        print(f"Loading weights from: {weights_path}")
        model = YOLO(weights_path)
        res = model.predict(img, imgsz=640, conf=0.35, verbose=False)[0]
        boxes = []
        for b in res.boxes:
            xy = b.xyxy[0].tolist()
            boxes.append(xy)
        
        if boxes:
            # Pick largest box
            boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box)
            print(f"YOLO found box: {x1},{y1},{x2},{y2}")
            img = img[y1:y2, x1:x2]
            if img.size == 0:
                print("Empty crop!")
                return
            cv2.imwrite("debug_yolo_crop.jpg", img)
        else:
            print("YOLO found no boxes, using full image.")
    except Exception as e:
        print(f"YOLO failed: {e}")

    ratio = img.shape[0] / 500.0
    orig = img.copy()
    img = cv2.resize(img, (int(img.shape[1]/ratio), 500))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try different Canny thresholds
    edged = cv2.Canny(blurred, 50, 200)
    cv2.imwrite("debug_edged.jpg", edged)
    
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    debug_cnt = img.copy()
    cv2.drawContours(debug_cnt, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite("debug_contours.jpg", debug_cnt)
    
    screenCnt = None
    for i, c in enumerate(cnts):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(f"Contour {i}: len(approx)={len(approx)}, area={cv2.contourArea(c)}")
        
        if len(approx) == 4:
            screenCnt = approx
            break
            
    if screenCnt is not None:
        print("Screen contour found!")
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
        cv2.imwrite("debug_screen_found.jpg", img)
    else:
        print("No screen contour found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_contours.py <image_path>")
    else:
        debug_contours(sys.argv[1])
