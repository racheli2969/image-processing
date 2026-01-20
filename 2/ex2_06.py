"""
Question 6: Effect of outliers on normalization
Shows how adding extreme pixel values (0 and 255) affects normalization results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ex2_04 import create_low_contrast_image
from ex2_05 import normalize, print_image_stats


def normalize_with_outliers():
    """
    Demonstrates the effect of outlier pixels on normalization
    """
    # Create a low contrast image
    original_img = create_low_contrast_image(fg=105, bg=100)
    
    # Normalize without outliers
    normalized_without_outliers = normalize(original_img)
    
    # Create a copy and add outliers
    img_with_outliers = original_img.copy()
    
    # Set one pixel to 0 (black)
    img_with_outliers[10, 10] = 0
    
    # Set one pixel to 255 (white)
    img_with_outliers[10, 20] = 255
    
    # Normalize with outliers
    normalized_with_outliers = normalize(img_with_outliers)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("COMPARISON: NORMALIZATION WITH AND WITHOUT OUTLIERS")
    print("=" * 70)
    
    print_image_stats(original_img, "Original Image (No Outliers)")
    print_image_stats(normalized_without_outliers, "Normalized (No Outliers)")
    
    print("\n--- AFTER ADDING OUTLIERS (one pixel = 0, one pixel = 255) ---")
    print_image_stats(img_with_outliers, "Image With Outliers")
    print_image_stats(normalized_with_outliers, "Normalized (With Outliers)")
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Without outliers
    axes[0, 0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Image\n(No Outliers)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(normalized_without_outliers, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Normalized\n(No Outliers)')
    axes[0, 1].axis('off')
    
    axes[0, 2].hist(normalized_without_outliers.ravel(), bins=256, range=[0, 256], 
                    color='blue', alpha=0.7)
    axes[0, 2].set_title('Histogram (No Outliers)')
    axes[0, 2].set_xlabel('Pixel Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: With outliers
    axes[1, 0].imshow(img_with_outliers, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Image With Outliers\n(2 pixels: 0 and 255)')
    axes[1, 0].axis('off')
    # Mark outlier locations
    axes[1, 0].plot(10, 10, 'r*', markersize=10, label='Pixel=0')
    axes[1, 0].plot(20, 10, 'y*', markersize=10, label='Pixel=255')
    axes[1, 0].legend()
    
    axes[1, 1].imshow(normalized_with_outliers, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Normalized\n(With Outliers)')
    axes[1, 1].axis('off')
    
    axes[1, 2].hist(normalized_with_outliers.ravel(), bins=256, range=[0, 256], 
                    color='red', alpha=0.7)
    axes[1, 2].set_title('Histogram (With Outliers)')
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print explanation
    print("\n" + "=" * 70)
    print("EXPLANATION OF THE EFFECT OF OUTLIERS")
    print("=" * 70)
    print("""
WITHOUT OUTLIERS:
- The original image has a narrow range: [100, 105]
- After normalization: stretched to full range [0, 255]
- The stretch factor is: 255 / (105 - 100) = 51
- This creates high contrast and makes details very visible

WITH OUTLIERS:
- After adding outliers, the range becomes: [0, 255]
- After normalization: already at [0, 255], minimal change
- The stretch factor is: 255 / (255 - 0) = 1
- The bulk of the image (values 100-105) gets compressed to a tiny range
- Result: The circle becomes nearly invisible!

WHY THIS HAPPENS:
The normalization formula is: normalized = (pixel - min) * 255 / (max - min)

Without outliers:
- normalized = (pixel - 100) * 255 / 5
- A pixel with value 100 → (100-100) * 51 = 0
- A pixel with value 105 → (105-100) * 51 = 255

With outliers:
- normalized = (pixel - 0) * 255 / 255 = pixel * 1
- A pixel with value 100 → (100-0) * 1 = 100
- A pixel with value 105 → (105-0) * 1 = 105

CONCLUSION:
Outliers drastically reduce the effectiveness of normalization!
The presence of just 2 extreme pixels prevents the normalization from
stretching the useful range of the image. This demonstrates why outlier
detection and removal can be important in image processing.

SOLUTION:
In practice, we might use percentile-based normalization (e.g., clip the
bottom 1% and top 1% of values) instead of using absolute min/max.
    """)
    print("=" * 70)


if __name__ == "__main__":
    normalize_with_outliers()
