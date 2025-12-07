from paddleocr import PaddleOCR
import cv2
import sys
import logging

# Suppress Paddle logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

def test_paddle(img_path):
    print(f"Testing PaddleOCR on {img_path}...")
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang='en') 
    
    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image")
        return

    # Run OCR
    result = ocr.ocr(img)
    
    print("Result Type:", type(result))
    print("Result:", result)
    if hasattr(result, 'rec_texts'):
        print("Rec Texts:", result.rec_texts)
    if isinstance(result, dict):
        print("Keys:", result.keys())
        if 'rec_texts' in result:
            print("Rec Texts:", result['rec_texts'])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_paddle.py <image_path>")
    else:
        test_paddle(sys.argv[1])
