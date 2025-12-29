
# use the Pillow library 
# a program that reads an image file and displays each color channel separately.
# the file name should be passed in the command line.
# usage example: python ex01.01.py my_image.jpg

import sys

import matplotlib.pyplot as plt
from PIL import Image


def main():
    # Check if filename is provided
    if len(sys.argv) < 2:
        print("Usage: python ex1.01.py <image_file>")
        sys.exit(1)

    # Read the image
    filename = sys.argv[1]
    try:
        img = Image.open(filename)
        # Convert to RGB if needed
        img = img.convert('RGB')
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Split into color channels
    r, g, b = img.split()

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Display original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display red channel
    axes[1].imshow(r, cmap='Reds')
    axes[1].set_title('Red Channel')
    axes[1].axis('off')

    # Display green channel
    axes[2].imshow(g, cmap='Greens')
    axes[2].set_title('Green Channel')
    axes[2].axis('off')

    # Display blue channel
    axes[3].imshow(b, cmap='Blues')
    axes[3].set_title('Blue Channel')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
