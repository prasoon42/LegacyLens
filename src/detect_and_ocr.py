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
def preprocess_image(img, max_side=1200):
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def clean_segment_text(text):
    # Map common 7-segment confusions
    # 7-seg specific replacements
    # Z -> 2, S -> 5, etc.
    subs = {
        'O': '0', 'o': '0', 'D': '0', 'Q': '0',
        'Z': '2', 'z': '2',
        'B': '8', 
        'A': '4', 'h': '4',
        'G': '6', 
        'T': '7', 
        'S': '5',
        'E': '3', 'F': '3', # Common 7-seg confusions
        '.': '', ' ': '',
        ']': '1', '[': '1', 'l': '1', 'I': '1'
    }
    for k, v in subs.items():
        text = text.replace(k, v)
    
    # Filter non-numeric? Or allow specific chars?
    # For now, keep alphanumeric but focus on digits
    return re.sub(r'[^0-9]', '', text)

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

    # Try multiple modes: Default, Inverted (for light-on-dark), and maybe Threshold
    modes = ["Default", "Inverted", "RedMask"]
    
    for mode in modes:
        print(f"[OCR] Running detection in mode: {mode}...", flush=True)
        img_input = enhance_image(base_img, mode)
        
        detected_in_pass = []
        
        result = ocr.ocr(img_input)
        
        # Fallback/Alternative: EasyOCR for RedMask if Paddle fails or just to augment
        if mode == 'RedMask':
             easy_reader = get_easyocr_reader()
             if easy_reader:
                 print(f"[OCR] Running EasyOCR on RedMask...", flush=True)
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
                if isinstance(text_tuple, (list, tuple)):
                    raw_items.append((box, text_tuple[0], text_tuple[1]))
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
        if m == 'RedMask': score = 3
        elif m == 'Inverted': score = 2
        else: score = 1
        return score

    # Before NMS, check for reverse pairs and boost the "correct" one
    # E.g., "123456" vs "954321" - prefer ascending
    def is_mostly_ascending(text):
        if len(text) < 2: return True
        ascending = sum(1 for i in range(len(text)-1) if text[i] <= text[i+1])
        return ascending >= len(text) / 2
    
    # Find overlapping pairs and adjust confidence
    for i in range(len(all_detected_items)):
        for j in range(i+1, len(all_detected_items)):
            item_a = all_detected_items[i]
            item_b = all_detected_items[j]
            
            iou = calculate_iou(item_a['box'], item_b['box'])
            if iou > 0.5:  # Significant overlap
                # Check if they're similar length (likely same display)
                if abs(len(item_a['text']) - len(item_b['text'])) <= 1:
                    # Prefer ascending order
                    a_ascending = is_mostly_ascending(item_a['text'])
                    b_ascending = is_mostly_ascending(item_b['text'])
                    
                    if a_ascending and not b_ascending:
                        # Boost A's confidence
                        item_a['conf'] = max(item_a['conf'], item_b['conf'] + 0.01)
                    elif b_ascending and not a_ascending:
                        # Boost B's confidence  
                        item_b['conf'] = max(item_b['conf'], item_a['conf'] + 0.01)
    
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
            if iou > 0.3: # Significant overlap
                # Collision! 
                # Since list is sorted by priority, A is better than B.
                # Drop B.
                dropped_indices.add(j)
                # print(f"Dropped {item_b['text']} ({item_b['mode']}) in favor of {item_a['text']} ({item_a['mode']})")
                
        if keep_a:
            indices_to_keep.append(i)
            
    final_items = [all_detected_items[i] for i in indices_to_keep]

    # Post-filter ghosts from final list if we have alternatives
    non_ghosts = [x for x in final_items if not (len(x['text'])>1 and x['text'].count('8')/len(x['text'])>0.6)]
    if non_ghosts:
        final_items = non_ghosts

    # Additional filter: Remove obvious UI noise
    # Keep only items that look like 7-segment readings:
    # - Purely numeric
    # - Reasonable length (3-6 digits for typical displays)
    # - Prefer ascending/sequential patterns
    segment_candidates = []
    for item in final_items:
        txt = item['text']
        # Must be numeric and reasonable length (most displays are 3-6 digits)
        if txt.isdigit() and 3 <= len(txt) <= 6:
            # Calculate pattern score
            pattern_score = 1.0
            
            # Heavily favor ascending sequences
            if is_mostly_ascending(txt):
                pattern_score *= 3.0
            
            # Penalize repeated digits (like "8888")
            unique_ratio = len(set(txt)) / len(txt)
            pattern_score *= unique_ratio
            
            # Final score: pattern × confidence × length
            item['score'] = pattern_score * item['conf'] * len(txt)
            segment_candidates.append(item)
    
    # If we found good candidates, use only the best one
    if segment_candidates:
        segment_candidates.sort(key=lambda x: x['score'], reverse=True)
        final_items = [segment_candidates[0]]
    
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
