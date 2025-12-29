from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from helper_functions import compute_histogram, stretch_histogram, load_image

def main():
    # a. Read an image file
    image_path = input("Enter the path to the image file: ")
    img = load_image(image_path)
    if img is None:
        return
    
    print(f"Image loaded successfully: {img.size[0]}x{img.size[1]} pixels")
    
    # b. Convert to black and white (grayscale)
    gray_img = img.convert('L')
    print("Image converted to grayscale")
        
     # c. Compute the histogram
    histogram_original = compute_histogram(gray_img)
    print("Histogram computed")        
    # d. Stretch the histogram and produce a new image
    stretched_img = stretch_histogram(gray_img)
    histogram_stretched = compute_histogram(stretched_img)
    print("Histogram stretched")
        
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))        
    # Original grayscale image
    axes[0, 0].imshow(gray_img, cmap='gray')        
    axes[0, 0].set_title('Original Grayscale Image')
    axes[0, 0].axis('off')
        
    # Original histogram
    axes[0, 1].bar(range(256), histogram_original, color='gray', width=1)
    axes[0, 1].set_title('Original Histogram')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
        
    # Stretched image
    axes[1, 0].imshow(stretched_img, cmap='gray')
    axes[1, 0].set_title('Stretched Histogram Image')
    axes[1, 0].axis('off')
        
    # Stretched histogram
    axes[1, 1].bar(range(256), histogram_stretched, color='gray', width=1)
    axes[1, 1].set_title('Stretched Histogram')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')   
        
    plt.tight_layout()
    plt.show()
    
    # Save the stretched image
    path = Path(image_path)
    output_path = path.parent / f"{path.stem}_stretched{path.suffix}"
    stretched_img.save(output_path)
    print(f"Stretched image saved to: {output_path}")

if __name__ == "__main__":
    main()