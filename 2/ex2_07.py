"""
Question 7: Compute and display histogram
Reads an image, converts to grayscale, computes histogram without library functions,
and displays using matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_histogram(img):
    """
    Computes histogram of a grayscale image without using library functions
    
    Parameters:
    img (numpy.ndarray): Grayscale image
    
    Returns:
    numpy.ndarray: Histogram array of size 256
    """
    # Initialize histogram array with 256 bins (for values 0-255)
    histogram = np.zeros(256, dtype=np.int32)
    
    # Get image dimensions
    height, width = img.shape
    
    # Count frequency of each pixel value
    for y in range(height):
        for x in range(width):
            pixel_value = img[y, x]
            histogram[pixel_value] += 1
    
    return histogram


def convert_to_grayscale(image_path):
    """
    Reads an image and converts it to grayscale
    
    Parameters:
    image_path (str): Path to the image file
    
    Returns:
    numpy.ndarray: Grayscale image
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    return gray_img


def display_histogram(histogram, title="Image Histogram"):
    """
    Displays histogram using matplotlib
    
    Parameters:
    histogram (numpy.ndarray): Histogram array
    title (str): Title for the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot histogram as bar chart
    plt.bar(range(256), histogram, width=1.0, edgecolor='none', color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main_with_sample_image():
    """
    Main function that uses a generated test image
    """
    # Since we may not have an image file, let's create a test image
    print("Creating a test grayscale image...")
    
    # Create a test image with various intensities
    from ex2_01 import create_gradient_image
    test_img = create_gradient_image(400, 400)
    
    # Add some shapes with different intensities
    cv2.circle(test_img, (100, 100), 50, 150, -1)
    cv2.rectangle(test_img, (200, 200), (350, 350), 80, -1)
    cv2.circle(test_img, (300, 100), 40, 200, -1)
    
    # Display the test image
    plt.figure(figsize=(8, 8))
    plt.imshow(test_img, cmap='gray')
    plt.title('Test Grayscale Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Compute histogram WITHOUT using library functions
    print("\nComputing histogram manually (without cv2.calcHist)...")
    histogram = compute_histogram(test_img)
    
    # Display the histogram
    display_histogram(histogram, "Histogram (Computed Manually)")
    
    # Print statistics
    print(f"\nHistogram Statistics:")
    print(f"Total pixels: {np.sum(histogram)}")
    print(f"Expected total: {test_img.shape[0] * test_img.shape[1]}")
    print(f"Number of unique values: {np.count_nonzero(histogram)}")
    print(f"Most common pixel value: {np.argmax(histogram)}")
    print(f"Frequency of most common value: {np.max(histogram)}")
    
    # For comparison, show cv2.calcHist result
    hist_cv2 = cv2.calcHist([test_img], [0], None, [256], [0, 256])
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(256), histogram, width=1.0, edgecolor='none', color='blue', alpha=0.7)
    plt.title('Manual Histogram Computation')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(256), hist_cv2.flatten(), width=1.0, edgecolor='none', color='red', alpha=0.7)
    plt.title('cv2.calcHist() Result (for comparison)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Verify that both methods give the same result
    if np.array_equal(histogram, hist_cv2.flatten().astype(np.int32)):
        print("\n✓ SUCCESS: Manual histogram matches cv2.calcHist() result!")
    else:
        print("\n✗ WARNING: Results differ!")


def main_with_file(image_path):
    """
    Main function that reads an image from file
    
    Parameters:
    image_path (str): Path to the image file
    """
    try:
        # Read and convert image to grayscale
        print(f"Reading image from: {image_path}")
        gray_img = convert_to_grayscale(image_path)
        
        # Display the grayscale image
        plt.figure(figsize=(8, 8))
        plt.imshow(gray_img, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Compute histogram
        print("Computing histogram...")
        histogram = compute_histogram(gray_img)
        
        # Display the histogram
        display_histogram(histogram, f"Histogram of {image_path}")
        
        # Print statistics
        print(f"\nImage Statistics:")
        print(f"Image size: {gray_img.shape}")
        print(f"Total pixels: {np.sum(histogram)}")
        print(f"Number of unique values: {np.count_nonzero(histogram)}")
        print(f"Min pixel value: {np.argmax(histogram > 0)}")
        print(f"Max pixel value: {255 - np.argmax(histogram[::-1] > 0)}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Using sample image instead...")
        main_with_sample_image()


if __name__ == "__main__":
    import sys
    
    # Check if an image path was provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        main_with_file(image_path)
    else:
        print("No image path provided. Using generated test image.")
        print("To use your own image, run: python ex2_07.py <path_to_image>")
        print("-" * 60)
        main_with_sample_image()
