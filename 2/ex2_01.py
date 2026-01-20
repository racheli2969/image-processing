"""
Question 1: Create a gradient image
Creates a grayscale image where pixel (0,0) is 0 (black) and pixel (width-1, height-1) is 255 (white)
with gradual color transition between corners.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_gradient_image(height, width):
    """
    Creates a grayscale gradient image
    
    Parameters:
    height (int): Height of the image
    width (int): Width of the image
    
    Returns:
    numpy.ndarray: Grayscale gradient image
    """
    # Create empty image
    img = np.zeros((height, width), dtype=np.uint8)
    
    # For each pixel, calculate its gradient value
    # The value should transition from 0 at (0,0) to 255 at (height-1, width-1)
    for y in range(height):
        for x in range(width):
            # Calculate the gradient value based on position
            # Normalize by the maximum distance (diagonal)
            value = (x + y) * 255 / (height - 1 + width - 1)
            img[y, x] = int(value)
    
    return img


if __name__ == "__main__":
    # Create gradient image
    gradient_img = create_gradient_image(255, 255)
    
    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(gradient_img, cmap='gray')
    plt.title('Gradient Image (255x255)')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Test with different sizes
    gradient_img_500 = create_gradient_image(500, 500)
    plt.figure(figsize=(8, 8))
    plt.imshow(gradient_img_500, cmap='gray')
    plt.title('Gradient Image (500x500)')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
