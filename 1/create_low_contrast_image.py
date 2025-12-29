"""Create a low-contrast test image to demonstrate histogram stretching"""

from PIL import Image
import numpy as np

# Create a low-contrast image (only using pixel values 50-150 instead of 0-255)
width, height = 400, 400
img_array = np.random.randint(50, 150, (height, width, 3), dtype=np.uint8)

# Add some shapes to make it more interesting
# Create a dark rectangle
img_array[100:200, 100:300] = np.random.randint(50, 80, (100, 200, 3), dtype=np.uint8)

# Create a lighter circle area
y, x = np.ogrid[:height, :width]
center_y, center_x = 300, 300
radius = 60
mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
img_array[mask] = np.random.randint(120, 150, (mask.sum(), 3), dtype=np.uint8)

# Save the low-contrast image
img = Image.fromarray(img_array)
img.save('../low_contrast_test.jpg')

print("Created low_contrast_test.jpg")
print(f"Pixel value ranges:")
print(f"  Red: {img_array[:,:,0].min()} to {img_array[:,:,0].max()}")
print(f"  Green: {img_array[:,:,1].min()} to {img_array[:,:,1].max()}")
print(f"  Blue: {img_array[:,:,2].min()} to {img_array[:,:,2].max()}")
print("\nThis image only uses a narrow range (50-150), not the full 0-255 range.")
print("Run: python ex1.03.py")
print("Then enter: ../low_contrast_test.jpg")
print("You'll see a clear difference after histogram stretching!")
