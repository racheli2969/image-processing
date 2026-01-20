"""
Question 5: Image Normalization
Implements normalization without using cv2.normalize function
Normalizes image so min value becomes 0 and max value becomes 255
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ex2_04 import create_low_contrast_image


def print_image_stats(img, title="Image Statistics"):
    """
    Prints min, max, mean pixel values and stretch factor
    
    Parameters:
    img (numpy.ndarray): Grayscale image
    title (str): Title for the output
    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
    mean_val = np.mean(img)
    
    print(f"\n=== {title} ===")
    print(f"Minimum pixel value: {min_val}")
    print(f"Maximum pixel value: {max_val}")
    print(f"Mean pixel value: {mean_val:.2f}")
    
    if max_val != min_val:
        stretch_factor = 255 / (max_val - min_val)
        print(f"Stretch factor (255/(max-min)): {stretch_factor:.4f}")
    else:
        print("Cannot compute stretch factor (max equals min)")
    print("=" * 50)


def normalize(src_image):
    """
    Normalizes image to range [0, 255] without using cv2.normalize
    
    Parameters:
    src_image (numpy.ndarray): Source grayscale image
    
    Returns:
    numpy.ndarray: Normalized image
    """
    # Get min and max values
    min_val, max_val, _, _ = cv2.minMaxLoc(src_image)
    
    # If all pixels have the same value, return the image as is
    if max_val == min_val:
        return src_image.copy()
    
    # Convert to float for precise calculations
    src_float = src_image.astype(np.float32)
    
    # Normalize: subtract min, then multiply by 255/(max-min)
    dst_float = (src_float - min_val) * (255.0 / (max_val - min_val))
    
    # Convert back to uint8 with clipping
    dst = np.clip(dst_float, 0, 255).astype(np.uint8)
    
    return dst


if __name__ == "__main__":
    # Create a low contrast image from question 4
    low_contrast_img = create_low_contrast_image(fg=105, bg=100)
    
    # Print statistics of original image
    print_image_stats(low_contrast_img, "Original Low Contrast Image")
    
    # Normalize the image
    normalized_img = normalize(low_contrast_img)
    
    # Print statistics of normalized image
    print_image_stats(normalized_img, "Normalized Image")
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    im0 = axes[0].imshow(low_contrast_img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Low Contrast Image\n(fg=105, bg=100)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])
    
    # Normalized image
    im1 = axes[1].imshow(normalized_img, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Normalized Image\n(stretched to [0, 255])')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Histograms comparison
    axes[2].hist(low_contrast_img.ravel(), bins=256, range=[0, 256], 
                 alpha=0.5, label='Original', color='blue')
    axes[2].hist(normalized_img.ravel(), bins=256, range=[0, 256], 
                 alpha=0.5, label='Normalized', color='red')
    axes[2].set_title('Histogram Comparison')
    axes[2].set_xlabel('Pixel Value')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nRESULTS:")
    print("The normalization process stretches the pixel values from the")
    print("narrow range [100, 105] to the full range [0, 255].")
    print("This significantly increases the contrast and makes the circle")
    print("much more visible in the image.")
