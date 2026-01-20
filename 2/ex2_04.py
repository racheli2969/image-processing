"""
Question 4: Create low contrast image
Creates an image with low contrast by using similar values for foreground and background
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_low_contrast_image(fg, bg, height=400, width=400):
    """
    Creates a low-contrast image with a circle on a background
    
    Parameters:
    fg (int): Foreground color value (0-255)
    bg (int): Background color value (0-255)
    height (int): Image height
    width (int): Image width
    
    Returns:
    numpy.ndarray: Low contrast grayscale image
    """
    # Create image filled with background color
    img = np.full((height, width), bg, dtype=np.uint8)
    
    # Draw a circle in the center with foreground color
    center = (width // 2, height // 2)
    radius = min(width, height) // 3
    
    cv2.circle(img, center, radius, int(fg), -1)  # -1 means filled circle
    
    return img


if __name__ == "__main__":
    # Example 1: Very low contrast (difference of 5)
    low_contrast_img1 = create_low_contrast_image(fg=105, bg=100)
    
    # Example 2: Slightly higher contrast (difference of 20)
    low_contrast_img2 = create_low_contrast_image(fg=120, bg=100)
    
    # Example 3: Medium contrast (difference of 50)
    low_contrast_img3 = create_low_contrast_image(fg=150, bg=100)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(low_contrast_img1, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Very Low Contrast\n(fg=105, bg=100, diff=5)')
    axes[0].axis('off')
    
    axes[1].imshow(low_contrast_img2, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Low Contrast\n(fg=120, bg=100, diff=20)')
    axes[1].axis('off')
    
    axes[2].imshow(low_contrast_img3, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Medium Contrast\n(fg=150, bg=100, diff=50)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("Low contrast images created successfully!")
    print(f"Image 1: fg={105}, bg={100}, contrast={abs(105-100)}")
    print(f"Image 2: fg={120}, bg={100}, contrast={abs(120-100)}")
    print(f"Image 3: fg={150}, bg={100}, contrast={abs(150-100)}")
