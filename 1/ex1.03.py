# use the Pillow library
# a program that:
# a reads an image file
# b computes the histogram for each color separately
# c performs "histogram stretching" for each color separately
# d produces and displays an image of the result

# usage example: python ex1.03.py my_image.jpg

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import compute_histogram, histogram_stretching, load_image

def main():
    # a) Read an image file
    image_path = input("Enter the path to the image file: ")
    img = load_image(image_path, mode='RGB')
    if img is None:
        return
    
    print(f"Image loaded successfully: {img.size[0]}x{img.size[1]} pixels")
    
    # Convert image to numpy array for processing
    img_array = np.array(img)
    
    # Split into color channels
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2]
    
    # b) Compute histogram for each color separately
    red_hist = compute_histogram(red_channel)
    green_hist = compute_histogram(green_channel)
    blue_hist = compute_histogram(blue_channel)
    
    print("\nHistogram computed for each color channel (R, G, B)")
    print(f"Red channel range: {np.min(red_channel)} to {np.max(red_channel)}")
    print(f"Green channel range: {np.min(green_channel)} to {np.max(green_channel)}")
    print(f"Blue channel range: {np.min(blue_channel)} to {np.max(blue_channel)}")
    
    # c) Perform histogram stretching for each color separately
    red_stretched = histogram_stretching(red_channel)
    green_stretched = histogram_stretching(green_channel)
    blue_stretched = histogram_stretching(blue_channel)
    
    print("\nHistogram stretching applied")
    print(f"Red channel stretched range: {np.min(red_stretched)} to {np.max(red_stretched)}")
    print(f"Green channel stretched range: {np.min(green_stretched)} to {np.max(green_stretched)}")
    print(f"Blue channel stretched range: {np.min(blue_stretched)} to {np.max(blue_stretched)}")
    
    # Combine stretched channels back into an image
    stretched_array = np.stack([red_stretched, green_stretched, blue_stretched], axis=2)
    stretched_img = Image.fromarray(stretched_array, 'RGB')
    
    # Compute histograms for stretched image
    red_hist_stretched = compute_histogram(red_stretched)
    green_hist_stretched = compute_histogram(green_stretched)
    blue_hist_stretched = compute_histogram(blue_stretched)
    
    # d) Display the original and result images with histograms
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Original histograms
    axes[0, 1].bar(range(256), red_hist, color='red', alpha=0.6, width=1)
    axes[0, 1].set_title('Red Channel Histogram (Original)')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[0, 2].bar(range(256), green_hist, color='green', alpha=0.6, width=1)
    axes[0, 2].set_title('Green Channel Histogram (Original)')
    axes[0, 2].set_xlabel('Pixel Value')
    axes[0, 2].set_ylabel('Frequency')
    
    axes[0, 3].bar(range(256), blue_hist, color='blue', alpha=0.6, width=1)
    axes[0, 3].set_title('Blue Channel Histogram (Original)')
    axes[0, 3].set_xlabel('Pixel Value')
    axes[0, 3].set_ylabel('Frequency')
    
    # Stretched image
    axes[1, 0].imshow(stretched_img)
    axes[1, 0].set_title('Histogram Stretched Image', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Stretched histograms
    axes[1, 1].bar(range(256), red_hist_stretched, color='red', alpha=0.6, width=1)
    axes[1, 1].set_title('Red Channel Histogram (Stretched)')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    axes[1, 2].bar(range(256), green_hist_stretched, color='green', alpha=0.6, width=1)
    axes[1, 2].set_title('Green Channel Histogram (Stretched)')
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Frequency')
    
    axes[1, 3].bar(range(256), blue_hist_stretched, color='blue', alpha=0.6, width=1)
    axes[1, 3].set_title('Blue Channel Histogram (Stretched)')
    axes[1, 3].set_xlabel('Pixel Value')
    axes[1, 3].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Optionally save the result
    save_option = input("\nDo you want to save the stretched image? (y/n): ")
    if save_option.lower() == 'y':
        output_path = input("Enter the output file path: ")
        stretched_img.save(output_path)
        print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()