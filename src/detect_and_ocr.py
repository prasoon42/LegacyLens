import os
import time
import re
import cv2
import numpy as np

# ----------------- Helpers: IoU / NMS / merging -----------------
def iou(a,b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interW = max(0, xB-xA)
    interH = max(0, yB-yA)
    inter = interW * interH
    areaA = max(0,(a[2]-a[0]))*max(0,(a[3]-a[1]))
    areaB = max(0,(b[2]-b[0]))*max(0,(b[3]-b[1]))
    union = areaA + areaB - inter
    if union<=0:
        return 0.0
    return inter/union

def nms_boxes(boxes, iou_thresh=0.35):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    used = [False]*len(boxes)
    for i,b in enumerate(boxes):
        if used[i]:
            continue
        keep.append(b)
        for j in range(i+1,len(boxes)):
            if used[j]:
                continue
            if iou(b, boxes[j]) > iou_thresh:
                used[j] = True
    return keep

# ----------------- PaddleOCR SETUP -----------------
_PADDLE_OCR = None

def get_reader():
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        try:
            from paddleocr import PaddleOCR
            # Initialize PaddleOCR
            _PADDLE_OCR = PaddleOCR(use_angle_cls=False, lang='en')
            print("[init] PaddleOCR initialized.", flush=True)
        except Exception as e:
            raise RuntimeError("Install paddlepaddle and paddleocr properly. Error: "+str(e))
    return _PADDLE_OCR

# ----------------- PREPROCESS -----------------
def preprocess_image(img, max_side=1000):
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def apply_preprocessing(img, mode="Default"):
    # Add padding to help with edge digits
    img_padded = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    if mode == "Default":
        return img_padded
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)
    if mode == "Grayscale":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "High Contrast":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    if mode == "Thresholding":
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    if mode == "Denoise":
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    return img

# ----------------- Local YOLO fallback -----------------
def local_yolo_detect(img, conf_thres=0.35, min_area=200):
    try:
        from ultralytics import YOLO
    except Exception:
        print("[YOLO fallback] ultralytics not installed; skipping local YOLO fallback.", flush=True)
        return []
    weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "yolov8n.pt")
    if not os.path.exists(weights_path):
        weights_path = "yolov8n.pt"
    try:
        print(f"[YOLO fallback] loading weights {weights_path} ...", flush=True)
        model = YOLO(weights_path)
        res = model.predict(img, imgsz=640, conf=conf_thres, verbose=False)[0]
        boxes = []
        for b in res.boxes:
            try:
                xy = b.xyxy[0].tolist()
            except Exception:
                continue
            x1,y1,x2,y2 = map(int, xy)
            conf = float(getattr(b, "conf", [0.0])[0]) if hasattr(b, "conf") else 0.0
            cls = int(getattr(b, "cls", [0])[0]) if hasattr(b, "cls") else 0
            area = (x2-x1)*(y2-y1)
            if area < min_area:
                continue
            boxes.append([x1,y1,x2,y2,conf,cls])
        print(f"[YOLO fallback] returned {len(boxes)} boxes", flush=True)
        return boxes
    except Exception as e:
        print("[YOLO fallback] error:", e, flush=True)
        return []

def detect_digits_raw(img):
    # Skip Roboflow for now as per logs
    print("[detect_digits_raw] Roboflow empty or failed; trying local YOLO...", flush=True)
    boxes = local_yolo_detect(img)
    return boxes or []

def postprocess_boxes(raw_boxes, min_conf=0.45, min_area=600, iou_thresh=0.35):
    if not raw_boxes:
        return []
    filtered = []
    for b in raw_boxes:
        x1,y1,x2,y2,conf,cls = b
        w = max(0, x2-x1); h = max(0, y2-y1)
        area = w*h
        if conf < min_conf:
            continue
        if area < min_area:
            continue
        filtered.append(b)
    filtered_nms = nms_boxes(filtered, iou_thresh=iou_thresh)
    return filtered_nms

def detect_digits(img):
    raw = detect_digits_raw(img)
    boxes = postprocess_boxes(raw, min_conf=0.30, min_area=1000, iou_thresh=0.35)
    if not boxes:
        H, W = img.shape[:2]
        return [[0, 0, W, H, 1.0, 0]]
    return boxes

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def map_rows_to_metrics(img, boxes):
    """
    PaddleOCR-based Screen-First OCR:
    1. Crop Device (YOLO).
    2. Find LCD Contour & Warp.
    3. Run PaddleOCR on Warped Image.
    4. Fallback to Center Crop if needed.
    """
    # DISABLED: YOLO keeps detecting wrong regions (labels instead of digits)
    # Force using full image for now
    H, W = img.shape[:2]
    device_box = [0, 0, W, H]
    print(f"[YOLO] Disabled - using full image.", flush=True)
    
    # if not boxes:
    #     H, W = img.shape[:2]
    #     device_box = [0, 0, W, H]
    # else:
    #     device_box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    #     # Check if box is too small (e.g. < 25% of image)
    #     x1, y1, x2, y2 = map(int, device_box[:4])
    #     box_area = (x2-x1)*(y2-y1)
    #     img_area = img.shape[0] * img.shape[1]
    #     if box_area < (img_area * 0.25):
    #         print(f"[YOLO] Box area {box_area} is too small (<25% of {img_area}). Using full image.", flush=True)
    #         H, W = img.shape[:2]
    #         device_box = [0, 0, W, H]
    
    x1, y1, x2, y2 = map(int, device_box[:4])
    H, W = img.shape[:2]
    pad = 10
    sx = max(0, x1 - pad); sy = max(0, y1 - pad)
    ex = min(W, x2 + pad); ey = min(H, y2 + pad)
    device_crop = img[sy:ey, sx:ex]
    
    if device_crop.size == 0: return {}, None

    # --- Step 2: Find LCD Contour & Warp ---
    ratio = device_crop.shape[0] / 500.0
    orig_crop = device_crop.copy()
    device_crop_small = cv2.resize(device_crop, (int(device_crop.shape[1]/ratio), 500))
    
    gray = cv2.cvtColor(device_crop_small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
            
    # DISABLED: Perspective warp causes too many issues with small/bad crops
    # Just use the original crop directly
    warped = orig_crop
    print("[Perspective] Using original crop (warp disabled).", flush=True)
    
    # if screenCnt is not None:
    #     warped = four_point_transform(orig_crop, screenCnt.reshape(4, 2) * ratio)
    #     # Validate aspect ratio and minimum size
    #     h_w, w_w = warped.shape[:2]
    #     if h_w > 0 and (w_w / h_w > 3.0 or h_w / w_w > 3.0):
    #         print(f"[Perspective] Warped aspect ratio {w_w/h_w:.2f} is extreme. Discarding warp.", flush=True)
    #         warped = orig_crop
    #     elif h_w < 200:  # Too small to OCR reliably
    #         print(f"[Perspective] Warped image too small ({h_w}px height). Using original crop.", flush=True)
    #         warped = orig_crop
    #     else:
    #         print("[Perspective] Screen contour found and warped.", flush=True)
    # else:
    #     warped = orig_crop
    #     print("[Perspective] No screen contour found, using full crop.", flush=True)

    target_h = 800
    
    # Helper to run PaddleOCR on an image
    def run_ocr_on_image(input_img):
        ocr = get_reader()
        # Upscale small crops for better OCR
        # scale_local = target_h / input_img.shape[0]
        # img_resized = cv2.resize(input_img, (int(input_img.shape[1]*scale_local), target_h))
        img_resized = input_img
        print(f"[OCR Debug] Running OCR on image shape: {img_resized.shape}", flush=True)
        
        try:
            print("[OCR Debug] Calling ocr.ocr()...", flush=True)
            result = ocr.ocr(img_resized)
            print(f"[OCR Debug] ocr.ocr() returned type: {type(result)}", flush=True)
            
            parsed_results = []
            
            # Check output format
            if isinstance(result, dict) and 'rec_texts' in result:
                texts = result['rec_texts']
                boxes = result['rec_boxes']
                scores = result['rec_scores']
                for box, text, score in zip(boxes, texts, scores):
                    parsed_results.append((box, text, score))
            elif isinstance(result, list) and result:
                # Check if it's a list of dicts (v3+ wrapper)
                if isinstance(result[0], dict) and 'rec_texts' in result[0]:
                    res_dict = result[0]
                    texts = res_dict['rec_texts']
                    boxes = res_dict['rec_boxes']
                    scores = res_dict['rec_scores']
                    for box, text, score in zip(boxes, texts, scores):
                        parsed_results.append((box, text, score))
                # Check if it's a list of lists (older format)
                elif isinstance(result[0], list):
                    for line in result[0]:
                        box = line[0]
                        text = line[1][0]
                        score = line[1][1]
                        parsed_results.append((box, text, score))
            
            print(f"[OCR Debug] Found {len(parsed_results)} items", flush=True)
            return parsed_results, img_resized
        except Exception as e:
            print(f"[OCR Debug] Failed: {e}", flush=True)
            return [], img_resized

    # Run on Warped first
    print("[OCR] Running on Warped Image...", flush=True)
    all_results, debug_img_used = run_ocr_on_image(warped)
    
    # Validation Helper
    def validate_results(results):
        valid = []
        for (bbox, text, conf) in results:
            clean = re.sub(r'[^0-9]', '', text)
            if not clean: continue
            if len(clean) > 3: continue # Filter long noise
            val = int(clean)
            if val < 30 or val > 300: continue # Filter out of range
            valid.append(clean)
        return valid

    # Check if we got valid results
    valid_vals = validate_results(all_results)
    
    # If failed, try Center Crop
    if not valid_vals:
        print("[OCR] No valid results. Retrying with Center Crop...", flush=True)
        h, w = orig_crop.shape[:2]
        
        # For BP monitors, digits are usually on the RIGHT side, labels on LEFT
        # So crop to the right 60% horizontally, center 60% vertically
        cy, cx = h // 2, w // 2
        ch = int(h * 0.6)
        cw = int(w * 0.6)
        
        # Shift crop to the right side
        y1 = max(0, cy - ch//2)
        y2 = min(h, cy + ch//2)
        x1 = max(0, w - cw)  # Start from right side
        x2 = w  # Go to edge
        
        center_crop = orig_crop[y1:y2, x1:x2]
        
        if center_crop.size > 0:
            all_results, debug_img_used = run_ocr_on_image(center_crop)
            valid_vals = validate_results(all_results)

    ocr_input = debug_img_used
        
    # Helper for 7-segment text cleaning
    def clean_segment_text(text):
        # Common substitutions for 7-segment/LCD displays
        # Removed 'S':'5' to avoid SYS -> 55
        subs = {
            'I': '1', 'l': '1', '|': '1', 'i': '1',
            'O': '0', 'o': '0', 'D': '0', 'Q': '0',
            'Z': '2', 'z': '2',
            'B': '8', # B often recognized as 8
            'A': '4', # A often recognized as 4
            'G': '6', # G often recognized as 6
            'T': '7', # T often recognized as 7
            '.': '', ' ': ''
        }
        # Apply substitutions
        for k, v in subs.items():
            text = text.replace(k, v)
        # Remove any remaining non-digits
        return re.sub(r'[^0-9]', '', text)

    # Filter & Cluster
    components = []
    print(f"[OCR Debug] Final results count: {len(all_results)}", flush=True)
    for (bbox, text, conf) in all_results:
        # Filter out labels explicitly
        if any(label in text.upper() for label in ['SYS', 'DIA', 'PULSE', 'RATE', 'BPM', 'MMHG']):
            print(f"    -> Ignored label text: '{text}'", flush=True)
            continue
            
        clean_text = clean_segment_text(text)
        print(f"  - Raw: '{text}' -> Clean: '{clean_text}' Conf: {conf:.2f}", flush=True)
        
        if not clean_text: continue
        
        # Paddle bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        pts = np.array(bbox)
        # print(f"DEBUG: bbox shape {pts.shape}, content {pts}", flush=True)
        if pts.ndim == 1:
            # If it's flat [x1,y1, x2,y2, ...], reshape
            if len(pts) == 8:
                pts = pts.reshape(4, 2)
            elif len(pts) == 4:
                # Maybe [x1, y1, x2, y2]?
                x1, y1, x2, y2 = pts
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        
        min_y = np.min(pts[:, 1]); max_y = np.max(pts[:, 1])
        min_x = np.min(pts[:, 0])
        h_box = max_y - min_y
        cy = (min_y + max_y) / 2
        
        # Filter tiny noise (< 2.5% of screen height to exclude tiny labels)
        # The actual digits should be larger/bolder than label text
        if h_box < (target_h * 0.025): 
            print(f"    -> Ignored (too small: {h_box:.1f} < {target_h*0.025:.1f})", flush=True)
            continue
        
        components.append({
            'text': clean_text,
            'cy': cy,
            'min_x': min_x,
            'h': h_box,
            'conf': conf
        })
        
    if not components: 
        print("[OCR Debug] No components found after filtering.", flush=True)
        return {}, ocr_input

    # Cluster lines
    components.sort(key=lambda c: c['cy'])
    lines = []
    curr = [components[0]]
    for c in components[1:]:
        if abs(c['cy'] - curr[-1]['cy']) < (target_h * 0.02):
            curr.append(c)
        else:
            lines.append(curr)
            curr = [c]
    lines.append(curr)
    
    final_lines = []
    print(f"[OCR Debug] Clustered into {len(lines)} lines:", flush=True)
    for i, line in enumerate(lines):
        line.sort(key=lambda c: c['min_x'])
        # Check each component individually
        line_vals = []
        for c in line:
            txt = c['text']
            if len(txt) > 3: continue
            if not txt.isdigit(): continue
            val = int(txt)
            # Allow lower values (down to 10) to catch partial reads like "18" for "118"
            # We can validate sanity later (e.g. SYS > DIA)
            if 10 <= val <= 300:
                line_vals.append({'text': txt, 'cy': c['cy'], 'min_x': c['min_x'], 'h': c['h']})
        
        if line_vals:
            print(f"  Line {i}: Found valid components: {[v['text'] for v in line_vals]}", flush=True)
            final_lines.extend(line_vals)
        else:
            print(f"  Line {i}: No valid components in {[c['text'] for c in line]}", flush=True)
             
    # Sort by CY (vertical order)
    final_lines.sort(key=lambda l: l['cy'])
    
    # Filter out values in the TOP portion of image (where labels like "SYS" are)
    # Device digits are usually in the CENTER/BOTTOM portion
    if final_lines and len(final_lines) > 3:
        img_height = target_h if target_h > 0 else 800
        # Remove items in top 20% of image
        filtered_lines = [l for l in final_lines if l['cy'] > img_height * 0.2]
        if len(filtered_lines) >= 3:
            final_lines = filtered_lines
            print(f"[PaddleOCR] Filtered out top 20% values", flush=True)
    
    # Filter out UI overlay text
    if final_lines:
        # If we have more than 3 values, take the 3 LARGEST (by height) ones
        # (device digits are always larger than scale numbers or UI overlay)
        if len(final_lines) > 3:
            # Sort by height (descending) and take top 3
            final_lines_sorted = sorted(final_lines, key=lambda l: l.get('h', 0), reverse=True)
            print(f"[PaddleOCR] Candidates sorted by height:", flush=True)
            for l in final_lines_sorted:
                print(f"  - '{l['text']}': h={l.get('h')}, x={l.get('min_x')}, y={l.get('cy')}", flush=True)
            
            final_lines_sorted = final_lines_sorted[:3]
            
            # Re-sort by vertical position for correct SYS/DIA/PULSE order
            final_lines_sorted.sort(key=lambda l: l['cy'])
            vals = [l['text'] for l in final_lines_sorted]
            print(f"[PaddleOCR] Filtered to largest 3 values: {vals}", flush=True)
        else:
            vals = [l['text'] for l in final_lines]
    else:
        vals = []
    
    print(f"[PaddleOCR] Found values: {vals}", flush=True)
    
    result = {}
    if len(vals) >= 1: result['SYS'] = vals[0]
    if len(vals) >= 2: result['DIA'] = vals[1]
    if len(vals) >= 3: result['PULSE'] = vals[2]
    
    return result, ocr_input

def reconstruct_reading(img, boxes):
    metrics, _ = map_rows_to_metrics(img, boxes)
    values = []
    if 'SYS' in metrics: values.append(metrics['SYS'])
    if 'DIA' in metrics: values.append(metrics['DIA'])
    if 'PULSE' in metrics: values.append(metrics['PULSE'])
    return ' | '.join(values)

def visualize(img, boxes, reading_str=None, show_conf=True):
    vis = img.copy()
    for b in boxes:
        x1,y1,x2,y2,conf,_ = b
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        if show_conf:
            cv2.putText(vis, f"{conf:.2f}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    if reading_str:
        cv2.putText(vis, f"Reading: {reading_str}", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    return vis

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect_and_ocr.py <image_path>")
        sys.exit(1)
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image:", img_path)
        sys.exit(1)
    img = preprocess_image(img)
    raw = detect_digits_raw(img)
    boxes = detect_digits(img)
    reading = reconstruct_reading(img, boxes)
    metrics, debug_img = map_rows_to_metrics(img, boxes)
    vis = visualize(img, boxes, reading)
    out_path = "output.jpg"
    cv2.imwrite(out_path, vis)
    if debug_img is not None:
        cv2.imwrite("debug_ocr_input.jpg", debug_img)
        print("Saved debug OCR input to debug_ocr_input.jpg")
    print("Detected Reading:", reading)
    print("Mapped metrics:", metrics)
    print("Saved visualization to", out_path)
