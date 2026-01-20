"""
Question 3: Compare np.add vs cv2.add
Apply brighten function with both methods and explain the differences
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ex2_01 import create_gradient_image
from ex2_02 import brighten


def compare_brighten_methods():
    """
    Compares the results of brightening using numpy.add vs cv2.add
    """
    # Create gradient image from question 1
    original_img = create_gradient_image(255, 255)
    
    # Brighten value - use a large value to see the difference
    b = 100
    
    # Apply brighten with numpy
    brightened_np = brighten(original_img, b, "np")
    
    # Apply brighten with cv2
    brightened_cv2 = brighten(original_img, b, "cv2")
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Gradient Image')
    axes[0, 0].axis('off')
    
    # Brightened with numpy
    im1 = axes[0, 1].imshow(brightened_np, cmap='gray')
    axes[0, 1].set_title(f'Brightened with numpy.add (+{b})')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Brightened with cv2
    im2 = axes[1, 0].imshow(brightened_cv2, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f'Brightened with cv2.add (+{b})')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Difference between the two methods
    diff = np.abs(brightened_np.astype(np.int16) - brightened_cv2.astype(np.int16))
    im3 = axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title('Absolute Difference (np vs cv2)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('c:/Users/brach/OneDrive/מסמכים/RACHELI/השלמה לתואר/image-processing/2/brighten_comparison.png')
    plt.show()
    
    # Print statistics
    print("=== Comparison of numpy.add vs cv2.add ===\n")
    print(f"Original image range: [{original_img.min()}, {original_img.max()}]")
    print(f"Brightness value added: {b}\n")
    
    print(f"numpy.add result range: [{brightened_np.min()}, {brightened_np.max()}]")
    print(f"cv2.add result range: [{brightened_cv2.min()}, {brightened_cv2.max()}]\n")
    
    print("EXPLANATION:")
    print("=" * 60)
    print("The key difference between numpy.add and cv2.add:")
    print()
    print("1. numpy.add (np.add):")
    print("   - Allows OVERFLOW: When pixel value + b > 255, it wraps around")
    print("   - Example: 200 + 100 = 300, which wraps to 44 (300 % 256)")
    print("   - This creates unexpected artifacts in the bright areas")
    print()
    print("2. cv2.add:")
    print("   - Uses SATURATION: Clips values to valid range [0, 255]")
    print("   - Example: 200 + 100 = 300, which is clipped to 255")
    print("   - This preserves image integrity without artifacts")
    print()
    print("CONCLUSION:")
    print("For image processing, cv2.add is generally preferred because it")
    print("maintains valid pixel values and prevents overflow artifacts.")
    print("=" * 60)


if __name__ == "__main__":
    compare_brighten_methods()
