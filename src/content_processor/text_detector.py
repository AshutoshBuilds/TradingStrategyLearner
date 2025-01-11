import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Any, Optional

class TextDetector:
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """Initialize the text detector.
        
        Args:
            tesseract_cmd (Optional[str]): Path to Tesseract executable
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        self._setup_preprocessing()

    def _setup_preprocessing(self):
        """Set up image preprocessing parameters."""
        self.preprocessing_params = {
            'blur_kernel': (5, 5),
            'threshold_block_size': 11,
            'threshold_c': 2
        }

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better text detection.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, self.preprocessing_params['blur_kernel'], 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.preprocessing_params['threshold_block_size'],
            self.preprocessing_params['threshold_c']
        )
        
        return thresh

    def detect_text(self, image: np.ndarray, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Detect and extract text from an image.
        
        Args:
            image (np.ndarray): Input image
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            List[Dict[str, Any]]: Detected text regions and content
        """
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Get OCR data
        ocr_data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        
        # Extract text regions with confidence above threshold
        results = []
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            # Skip empty text
            if not ocr_data['text'][i].strip():
                continue
                
            # Get confidence
            conf = float(ocr_data['conf'][i])
            if conf < min_confidence * 100:  # Tesseract confidence is 0-100
                continue
                
            # Get bounding box
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            results.append({
                'text': ocr_data['text'][i],
                'confidence': conf / 100,  # Normalize to 0-1
                'bbox': (x, y, w, h),
                'page_num': ocr_data['page_num'][i],
                'block_num': ocr_data['block_num'][i],
                'par_num': ocr_data['par_num'][i],
                'line_num': ocr_data['line_num'][i],
                'word_num': ocr_data['word_num'][i]
            })
            
        return results

    def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text regions in an image without OCR.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict[str, Any]]: Detected text regions
        """
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Find contours
        contours, _ = cv2.findContours(
            processed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and process contours
        results = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w < 10 or h < 10:
                continue
                
            # Calculate region properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            results.append({
                'bbox': (x, y, w, h),
                'area': area,
                'perimeter': perimeter,
                'aspect_ratio': aspect_ratio,
                'contour': contour.tolist()
            })
            
        return results

    def extract_text_from_region(self, image: np.ndarray, region: Dict[str, Any],
                               min_confidence: float = 0.5) -> Optional[Dict[str, Any]]:
        """Extract text from a specific region in an image.
        
        Args:
            image (np.ndarray): Input image
            region (Dict[str, Any]): Region information
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            Optional[Dict[str, Any]]: Extracted text and confidence
        """
        # Extract region from image
        x, y, w, h = region['bbox']
        roi = image[y:y+h, x:x+w]
        
        # Preprocess region
        processed = self._preprocess_image(roi)
        
        # Get OCR data
        ocr_data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        
        # Combine all text with sufficient confidence
        text_parts = []
        total_conf = 0
        n_parts = 0
        
        for i, text in enumerate(ocr_data['text']):
            if not text.strip():
                continue
                
            conf = float(ocr_data['conf'][i])
            if conf < min_confidence * 100:
                continue
                
            text_parts.append(text)
            total_conf += conf
            n_parts += 1
            
        if not text_parts:
            return None
            
        return {
            'text': ' '.join(text_parts),
            'confidence': (total_conf / n_parts) / 100 if n_parts > 0 else 0
        } 