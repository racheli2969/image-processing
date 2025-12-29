"""Helper functions for image processing exercises"""

import numpy as np
from PIL import Image


def compute_histogram(channel_data):
    """
    Compute histogram for a single color channel or grayscale image
    
    Args:
        channel_data: numpy array or PIL Image object
        
    Returns:
        list: Histogram with 256 bins (0-255)
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(channel_data, Image.Image):
        channel_data = np.array(channel_data)
    
    # Use numpy for fast histogram computation
    histogram, _ = np.histogram(channel_data.flatten(), bins=256, range=(0, 256))
    return histogram.tolist()


def histogram_stretching(channel_data):
    """
    Perform histogram stretching on a single channel
    
    Args:
        channel_data: numpy array of pixel values
        
    Returns:
        numpy array: Stretched channel data with values in range 0-255
    """
    # Find minimum and maximum pixel values
    min_val = np.min(channel_data)
    max_val = np.max(channel_data)
    
    # Avoid division by zero
    if max_val == min_val:
        return channel_data
    
    # Apply histogram stretching formula: new_val = (old_val - min) * 255 / (max - min)
    stretched = ((channel_data - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
    return stretched


def stretch_histogram(image):
    """
    Stretch histogram of a grayscale PIL Image to use full range 0-255
    
    Args:
        image: PIL Image object (grayscale)
        
    Returns:
        PIL Image: New image with stretched histogram
    """
    # Convert to numpy array for efficient operations
    img_array = np.array(image)
    
    # Use histogram_stretching function (numpy-based)
    stretched_array = histogram_stretching(img_array)
    
    # Convert back to PIL Image
    return Image.fromarray(stretched_array, 'L')


def load_image(image_path, mode=None):
    """
    Load an image file with error handling
    
    Args:
        image_path: Path to the image file
        mode: Optional conversion mode ('RGB', 'L', etc.)
        
    Returns:
        PIL Image object or None if error
    """
    try:
        img = Image.open(image_path)
        if mode:
            img = img.convert(mode)
        return img
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
