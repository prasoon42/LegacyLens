import os
import cv2
import numpy as np
import re
from paddleocr import PaddleOCR
import easyocr

# ----------------- PaddleOCR SINGLETON -----------------
_PADDLE_OCR = None

def get_reader():
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        try:
            # Initialize PaddleOCR (English, no angle classification for speed if not needed)
            _PADDLE_OCR = PaddleOCR(use_angle_cls=False, lang='en')
            print("[init] PaddleOCR initialized.", flush=True)
        except Exception as e:
            raise RuntimeError("Install paddlepaddle and paddleocr properly. Error: "+str(e))
    return _PADDLE_OCR

_EASY_OCR = None
def get_easyocr_reader():
    global _EASY_OCR
    if _EASY_OCR is None:
        try:
            _EASY_OCR = easyocr.Reader(['en'])
            print("[init] EasyOCR initialized.", flush=True)
        except Exception as e:
            print(f"Warning: EasyOCR init failed: {e}")
    return _EASY_OCR

# ----------------- PREPROCESS -----------------
def preprocess_image(img, max_side=1600):
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def clean_segment_text(text):
    # Filter to keep alphanumeric, dots, and basic symbols first
    text = re.sub(r'[^a-zA-Z0-9.%°/.-]', '', text)
    
    # Only apply 7-segment substitutions if the string already contains a digit
    # or if it's a very common misread pattern. This preserves units like "PSI".
    if any(c.isdigit() for c in text) or len(text) <= 1:
        subs = {
            'O': '0', 'D': '0', 'U': '0',
            'I': '1', 'L': '1', '|': '1',
            'Z': '2',
            'S': '5', 's': '5',
            'G': '6', 'b': '6',
            'B': '8',
            'q': '9', 'g': '9'
        }
        for k, v in subs.items():
            text = text.replace(k, v)
    
    return text

def calculate_iou(box1, box2):
    # Simple fitting AABB IoU for robustness
    # box is list of [x,y]
    p1 = np.array(box1)
    p2 = np.array(box2)
    
    if p1.ndim == 1: p1 = p1.reshape((-1, 2))
    if p2.ndim == 1: p2 = p2.reshape((-1, 2))
    
    x1_min, y1_min = np.min(p1, axis=0)
    x1_max, y1_max = np.max(p1, axis=0)
    
    x2_min, y2_min = np.min(p2, axis=0)
    x2_max, y2_max = np.max(p2, axis=0)
    
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_width = max(0, xi_max - xi_min)
    inter_height = max(0, yi_max - yi_min)
    inter_area = inter_width * inter_height
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    union = area1 + area2 - inter_area
    if union <= 0: return 0
    return inter_area / union

def enhance_image(img, mode="Default"):
    if mode == "Inverted":
        return cv2.bitwise_not(img)
    elif mode == "Gray":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == "Contrast":
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    elif mode == "Denoised":
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    elif mode == "Binarized":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Denoise before binarization
        gray = cv2.medianBlur(gray, 3)
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    elif mode == "Sharpened":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(img, -1, kernel)
    elif mode == "Threshold":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    elif mode == "RedMask":
        # HSV Masking for Red LED
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Red wraps around 0/180 in HSV
        # Lower red range
        lower_red1 = np.array([0, 50, 100])
        upper_red1 = np.array([15, 255, 255])
        
        # Upper red range
        lower_red2 = np.array([165, 50, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        # DILATE to thicken segments
        kernel_dilate = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        # Return as BGR (White digits on black background)
        # Add padding (black border)
        mask_padded = cv2.copyMakeBorder(mask, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)
        
        # Resize to reasonable height for OCR (e.g. 100px)
        h, w = mask_padded.shape[:2]
        target_h = 100
        scale = target_h / h
        target_w = int(w * scale)
        mask_resized = cv2.resize(mask_padded, (target_w, target_h))
        
        return cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    elif mode == "AmberMask":
        # HSV Masking for Yellow/Orange LED
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Amber/Yellow/Orange range
        lower_amber = np.array([10, 50, 100])
        upper_amber = np.array([45, 255, 255])
        
        mask = cv2.inRange(hsv, lower_amber, upper_amber)
        
        # DILATE to thicken segments
        kernel_dilate = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        # Return as BGR (White digits on black background)
        mask_padded = cv2.copyMakeBorder(mask, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)
        
        h, w = mask_padded.shape[:2]
        target_h = 100
        scale = target_h / h
        target_w = int(w * scale)
        mask_resized = cv2.resize(mask_padded, (target_w, target_h))
        
        return cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    return img

def detect_text(img):
    """
    Generic 7-segment detection with Multi-Scale/Multi-Mode fallback.
    """
    ocr = get_reader()
    
    # 1. Preprocess (resize if too huge)
    base_img = preprocess_image(img)
    if base_img is None: return []

    all_detected_items = []
    seen_texts = set()

    # Try multiple modes for maximum accuracy
    modes = ["Default", "Inverted", "RedMask", "AmberMask", "Contrast", "Binarized", "Sharpened", "Denoised"]
    
    for mode in modes:
        print(f"[OCR] Running detection in mode: {mode}...", flush=True)
        img_input = enhance_image(base_img, mode)
        
        detected_in_pass = []
        
        result = ocr.ocr(img_input)
        
        # Fallback/Alternative: EasyOCR for Red/AmberMask if Paddle fails or just to augment
        if mode in ['RedMask', 'AmberMask']:
             easy_reader = get_easyocr_reader()
             if easy_reader:
                 print(f"[OCR] Running EasyOCR on {mode}...", flush=True)
                 easy_results = easy_reader.readtext(img_input)
                 # Convert EasyOCR results to Paddle format for consistent processing
                 # EasyOCR: ([[x,y]...], text, conf)
                 for (bbox, text, conf) in easy_results:
                     # EasyOCR bbox is list of 4 points
                     # Paddle expects list of list of points?
                     # We handle it in the loop below
                     detected_in_pass.append({
                        'text': clean_segment_text(text),
                        'conf': float(conf),
                        'box': bbox,
                        'mode': mode + "_EasyOCR"
                     })
        
        # detected_in_pass is already initialized
        
        # Handle PaddleOCR output formats
        if result is None or not result: continue
        
        # Check for empty result [None]
        if isinstance(result, list) and len(result) == 1 and result[0] is None:
            continue

        # Flatten logic
        content = result[0] if isinstance(result, list) and len(result) > 0 else result
        
        # Parse based on structure
        raw_items = []
        
        # Dict format (v3)
        if isinstance(content, dict) and 'rec_texts' in content:
            texts = content.get('rec_texts', [])
            boxes = content.get('rec_boxes', [])
            scores = content.get('rec_scores', [])
            for i, text in enumerate(texts):
                raw_items.append((boxes[i], text, scores[i]))
        # List of Lists format
        elif isinstance(content, list):
            for line in content:
                if not line or len(line) < 2: continue
                box = line[0]
                text_tuple = line[1]
                if isinstance(text_tuple, (list, tuple)) and len(text_tuple) >= 2:
                    raw_items.append((box, text_tuple[0], text_tuple[1]))
                elif isinstance(text_tuple, (list, tuple)) and len(text_tuple) == 1:
                    raw_items.append((box, text_tuple[0], 1.0))
                else:
                    raw_items.append((box, str(text_tuple), 1.0))
        
        # Process Raw Items
        for (box, text_raw, conf) in raw_items:
            # Clean text
            text_clean = clean_segment_text(text_raw)
            if not text_clean: continue
            
            # De-duplication key (text + approx location)
            # Simple dedup: just text value for now to avoid noisy duplicates across modes?
            # Better: Keep all valid ones and let user decide?
            # Or dedup by overlap.
            
            # Let's just key by text to avoid exact duplicates
            if text_clean in seen_texts: continue
            seen_texts.add(text_clean)
            
            print(f"  [Item] Raw: '{text_raw}' -> Clean: '{text_clean}' ({conf:.2f})")
            
            # Ensure box is a list for JSON serialization
            if isinstance(box, np.ndarray):
                box = box.tolist()
            elif isinstance(box, list):
                # Check if elements are numpy types
                box = [b.tolist() if isinstance(b, np.ndarray) else b for b in box]

            detected_in_pass.append({
                'text': text_clean,
                'conf': float(conf),
                'box': box,
                'mode': mode  # Store mode for priority
            })

        # Merge results instead of hard replacement
        all_detected_items.extend(detected_in_pass)
        
    # NMS Deduplication
    final_items = []
    
    # Sort potential items by Preference: RedMask > Inverted > Default
    # Within mode, by confidence?
    # Let's assign score: RedMask=3, Inverted=2, Default=1
    def get_priority(item):
        m = item.get('mode', 'Default')
        score = 0
        if m == 'RedMask': score = 4
        elif m == 'AmberMask': score = 3
        elif m == 'Inverted': score = 2
        else: score = 1
        return score

    # Removed is_mostly_ascending heuristic as it's too restrictive for general use
    
    # Find overlapping pairs and adjust confidence
    for i in range(len(all_detected_items)):
        for j in range(i+1, len(all_detected_items)):
            item_a = all_detected_items[i]
            item_b = all_detected_items[j]
            
            iou = calculate_iou(item_a['box'], item_b['box'])
            # Overlap check remains, but without ascending boost
            if iou > 0.5:  # Significant overlap
                # Just keep the one with higher confidence
                if item_a['conf'] < item_b['conf']:
                    item_a['conf'] = item_b['conf'] # Placeholder for sorting
    
    # Sort by Priority DESC, then Confidence DESC
    all_detected_items.sort(key=lambda x: (get_priority(x), x['conf']), reverse=True)
    
    indices_to_keep = []
    dropped_indices = set()
    
    for i in range(len(all_detected_items)):
        if i in dropped_indices: continue
        
        item_a = all_detected_items[i]
        
        # Ghost check again (just in case)
        txt = item_a['text']
        if len(txt) > 1 and txt.count('8') / len(txt) > 0.6:
             # Ghost. Only keep if it's the ONLY thing we have.
             # But here we are iterating sorted list.
             # If we have *any* preserved item that is NOT ghost, drop this.
             pass # Logic captured by lower priority or confidence usually?
             # Let's explicit drop if we have better candidates
             # But harder to do inside this loop.
             pass

        keep_a = True
        
        for j in range(i + 1, len(all_detected_items)):
            if j in dropped_indices: continue
            
            item_b = all_detected_items[j]
            
            iou = calculate_iou(item_a['box'], item_b['box'])
            if iou > 0.4: # Significant overlap
                # Collision! 
                # Instead of dropping, let's add it as an alternative if the text is different
                if item_a['text'] != item_b['text']:
                    if 'alternatives' not in item_a:
                        item_a['alternatives'] = []
                    if item_b['text'] not in item_a['alternatives']:
                        item_a['alternatives'].append(item_b['text'])
                
                # Still drop B from the main list so we don't have duplicate boxes
                dropped_indices.add(j)
                
        if keep_a:
            indices_to_keep.append(i)
            
    final_items = [all_detected_items[i] for i in indices_to_keep]

    # Post-filter ghosts from final list if we have alternatives
    non_ghosts = [x for x in final_items if not (len(x['text'])>1 and x['text'].count('8')/len(x['text'])>0.6)]
    if non_ghosts:
        final_items = non_ghosts

    # Keep all items that have reasonable confidence and aren't just noise
    diverse_items = []
    for item in final_items:
        txt = item['text']
        # Keep if it has at least one alphanumeric char and isn't too short unless it's a known unit
        if len(txt) >= 1 and any(c.isalnum() for c in txt):
            diverse_items.append(item)
    if diverse_items:
        final_items = diverse_items
    
    # Add spatial context (normalized coordinates and size)
    h, w = base_img.shape[:2]
    for item in final_items:
        pts = np.array(item['box'])
        if pts.ndim == 1:
            pts = pts.reshape((-1, 2))
            
        xmin, xmax = np.min(pts[:, 0]), np.max(pts[:, 0])
        ymin, ymax = np.min(pts[:, 1]), np.max(pts[:, 1])
        
        center_x = (xmin + xmax) / 2 / w
        center_y = (ymin + ymax) / 2 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h
        
        item['spatial'] = {
            'center': [round(center_x, 3), round(center_y, 3)],
            'size': [round(width, 3), round(height, 3)],
            'area': round(width * height, 4)
        }
    
    all_detected_items = final_items
    
    # Sort by vertical position (top to bottom), then horizontal (left to right) for display
    def get_pos(item):
        pts = np.array(item['box'])
        if pts.ndim == 1:
            pts = pts.reshape((-1, 2))
        cy = np.mean(pts[:, 1])
        cx = np.mean(pts[:, 0])
        return (int(cy // 20), int(cx)) # Row binning (20px) then Column

    all_detected_items.sort(key=get_pos)
    
    return all_detected_items

def visualize(img, items):
    vis = img.copy()
    
    # Ensure visualization matches preprocessing? 
    # For simplicity, we assume 'items' coordinates map to 'img' dimensions roughly if aspect ratio preserved
    # But since we resized in 'preprocess_image', coords are in resized space.
    # To confirm visualization accuracy, we should really visualize on the preprocessed image.
    
    img_processed = preprocess_image(img)
    vis = img_processed.copy()
    
    for item in items:
        box = np.array(item['box']).astype(np.int32)
        if box.ndim == 1:
            box = box.reshape((-1, 2))
        # Reshape to (N, 1, 2) for polylines
        box = box.reshape((-1, 1, 2))
        
        cv2.polylines(vis, [box], True, (0, 255, 0), 2)
        
        # Draw text
        x, y = box[0][0]
        cv2.putText(vis, f"{item['text']} ({item['conf']:.2f})", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                   
    return vis

if __name__ == "__main__":
    import sys
    # Test script in main
    pass
