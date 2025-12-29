# use the Pillow library
# a program that:
# a reads an image file
# b computes the histogram for each color separately
# c performs "histogram stretching" for each color separately
# d produces and displays an image of the result

from PIL import Image
import numpy as np
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
    
    # d) Display the original and result images
    img.show(title="Original Image")
    stretched_img.show(title="Histogram Stretched Image")
    
    # Optionally save the result
    save_option = input("\nDo you want to save the stretched image? (y/n): ")
    if save_option.lower() == 'y':
        output_path = input("Enter the output file path: ")
        stretched_img.save(output_path)
        print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()