import numpy as np
import cv2
from PIL import Image

def preprocess_image(image_data):
    """
    Preprocess the image from the canvas for model prediction
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        Raw image data from the canvas
        
    Returns:
    --------
    numpy.ndarray
        Preprocessed image ready for prediction
    """
    # Convert to grayscale if the image has color channels
    if len(image_data.shape) > 2:
        # The image from canvas has RGBA channels
        # Extract just the RGB part for grayscale conversion
        rgb_image = image_data[:, :, :3]
        
        # Convert RGB to grayscale
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_data
    
    # Invert the image (white digit on black background)
    # This depends on how the canvas is set up, sometimes we need to invert
    if np.mean(gray_image) > 127:  # If the background is bright
        gray_image = 255 - gray_image  # Invert
    
    # Apply thresholding to make it binary
    _, binary_image = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)
    
    # Find contours to center the digit
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the bounding box of the digit
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        
        # Add padding around the digit
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray_image.shape[1] - x, w + 2*padding)
        h = min(gray_image.shape[0] - y, h + 2*padding)
        
        # Crop the image to the bounding box
        digit_image = binary_image[y:y+h, x:x+w]
        
        # Calculate the larger dimension and create a square image
        larger_dim = max(w, h)
        square_image = np.zeros((larger_dim, larger_dim), dtype=np.uint8)
        
        # Center the digit in the square image
        offset_x = (larger_dim - w) // 2
        offset_y = (larger_dim - h) // 2
        square_image[offset_y:offset_y+h, offset_x:offset_x+w] = digit_image
        
        # Resize to 20x20 to match MNIST (keeping the digit centered with some border)
        resized_image = cv2.resize(square_image, (20, 20), interpolation=cv2.INTER_AREA)
        
        # Pad with 4 pixels on each side to get a 28x28 image
        padded_image = np.pad(resized_image, 4, mode='constant')
    else:
        # If no contours found, just resize the binary image
        padded_image = cv2.resize(binary_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to be between 0 and 1
    normalized_image = padded_image.astype('float32') / 255.0
    
    return normalized_image
