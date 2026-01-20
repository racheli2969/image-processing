"""
Question 2: Brighten function
Creates a function that adds a value to all pixels using either numpy.add or cv2.add
"""

import cv2
import numpy as np


def brighten(img, b, func):
    """
    Adds a constant value to all pixels in the image
    
    Parameters:
    img (numpy.ndarray): Grayscale image
    b (int): Value to add to all pixels
    func (str): Either "np" (for numpy.add) or "cv2" (for cv2.add)
    
    Returns:
    numpy.ndarray: Brightened image
    """
    if func == "np":
        # numpy.add allows overflow/underflow
        result = np.add(img, b)
    elif func == "cv2":
        # cv2.add performs saturation (clips values at 0 and 255)
        result = cv2.add(img, np.uint8(b))
    else:
        raise ValueError("func parameter must be either 'np' or 'cv2'")
    
    return result


if __name__ == "__main__":
    # Example usage
    from ex2_01 import create_gradient_image
    import matplotlib.pyplot as plt
    
    # Create a test image
    test_img = create_gradient_image(255, 255)
    
    # Test with numpy
    brightened_np = brighten(test_img, 50, "np")
    
    # Test with cv2
    brightened_cv2 = brighten(test_img, 50, "cv2")
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(test_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(brightened_np, cmap='gray')
    axes[1].set_title('Brightened with numpy.add')
    axes[1].axis('off')
    
    axes[2].imshow(brightened_cv2, cmap='gray')
    axes[2].set_title('Brightened with cv2.add')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
