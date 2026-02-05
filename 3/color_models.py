"""
Color Models Converter
======================
This program converts RGB color values to HSV, HSL, and YCrCb color models.
It implements both manual calculations and OpenCV's built-in functions for comparison.

Usage:
    python color_models.py R G B
    
Example:
    python color_models.py 255 128 64
"""

import sys
import cv2
import numpy as np


def rgb_to_hsv_manual(r, g, b):
    """
    Convert RGB to HSV color space using manual formula implementation.
    
    Args:
        r (int): Red component (0-255)
        g (int): Green component (0-255)
        b (int): Blue component (0-255)
    
    Returns:
        tuple: (h, s, v) where:
            h: Hue (0-360 degrees)
            s: Saturation (0-1)
            v: Value (0-1)
    
    Formula based on: https://en.wikipedia.org/wiki/HSL_and_HSV
    """
    # Normalize RGB values to [0, 1]
    r, g, b = r/255, g/255, b/255
    
    # Find maximum and minimum values
    cmax, cmin = max(r, g, b), min(r, g, b)
    delta = cmax - cmin

    # Calculate Hue
    if delta == 0:
        h = 0
    elif cmax == r:
        h = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        h = 60 * (((b - r) / delta) + 2)
    else:  # cmax == b
        h = 60 * (((r - g) / delta) + 4)

    # Calculate Saturation
    s = 0 if cmax == 0 else delta / cmax
    
    # Calculate Value
    v = cmax
    
    return h, s, v


def rgb_to_hsl_manual(r, g, b):
    """
    Convert RGB to HSL color space using manual formula implementation.
    
    Args:
        r (int): Red component (0-255)
        g (int): Green component (0-255)
        b (int): Blue component (0-255)
    
    Returns:
        tuple: (h, s, l) where:
            h: Hue (0-360 degrees)
            s: Saturation (0-1)
            l: Lightness (0-1)
    
    Formula based on: https://en.wikipedia.org/wiki/HSL_and_HSV
    """
    # Normalize RGB values to [0, 1]
    r, g, b = r/255, g/255, b/255
    
    # Find maximum and minimum values
    cmax, cmin = max(r, g, b), min(r, g, b)
    delta = cmax - cmin

    # Calculate Lightness
    l = (cmax + cmin) / 2

    # Calculate Hue and Saturation
    if delta == 0:
        h = s = 0
    else:
        # Calculate Saturation based on lightness
        s = delta / (1 - abs(2 * l - 1))
        
        # Calculate Hue
        if cmax == r:
            h = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60 * (((b - r) / delta) + 2)
        else:  # cmax == b
            h = 60 * (((r - g) / delta) + 4)

    return h, s, l


def rgb_to_ycrcb_manual(r, g, b):
    """
    Convert RGB to YCrCb color space using manual formula implementation.
    
    Args:
        r (int): Red component (0-255)
        g (int): Green component (0-255)
        b (int): Blue component (0-255)
    
    Returns:
        tuple: (y, cr, cb) where:
            y: Luma component (0-255)
            cr: Red-difference chroma component (0-255)
            cb: Blue-difference chroma component (0-255)
    
    Formula based on ITU-R BT.601 standard
    """
    # Calculate Y (luma) component
    y = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Calculate Cr (red-difference chroma) component
    cr = (r - y) * 0.713 + 128
    
    # Calculate Cb (blue-difference chroma) component
    cb = (b - y) * 0.564 + 128
    
    return y, cr, cb


if __name__ == "__main__":
    # Validate command line arguments
    if len(sys.argv) < 4:
        print("Usage: python color_models.py R G B")
        print("Example: python color_models.py 255 128 64")
        sys.exit(1)
    
    # Parse RGB values from command line
    R, G, B = map(int, sys.argv[1:4])
    
    # Validate RGB range
    if not all(0 <= val <= 255 for val in [R, G, B]):
        print("Error: RGB values must be in range 0-255")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Color Conversion Results for RGB({R}, {G}, {B})")
    print(f"{'='*60}\n")
    
    # ============ Manual Implementations ============
    print("Manual Implementations (Direct Formula):")
    print("-" * 60)
    
    h, s, v = rgb_to_hsv_manual(R, G, B)
    print(f"HSV:   H={h:.2f}°, S={s:.4f}, V={v:.4f}")
    
    h, s, l = rgb_to_hsl_manual(R, G, B)
    print(f"HSL:   H={h:.2f}°, S={s:.4f}, L={l:.4f}")
    
    y, cr, cb = rgb_to_ycrcb_manual(R, G, B)
    print(f"YCrCb: Y={y:.2f}, Cr={cr:.2f}, Cb={cb:.2f}")
    
    # ============ OpenCV Implementations ============
    print("\nOpenCV Built-in Functions:")
    print("-" * 60)
    
    # Create a single pixel image in BGR format (OpenCV uses BGR, not RGB)
    img = np.uint8([[[B, G, R]]])
    
    # Convert using OpenCV functions
    hsv_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[0][0]
    print(f"HSV:   H={hsv_cv[0]*2}°, S={hsv_cv[1]/255:.4f}, V={hsv_cv[2]/255:.4f}")
    print(f"       (Raw OpenCV values: {hsv_cv})")
    
    hls_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[0][0]
    print(f"HLS:   H={hls_cv[0]*2}°, L={hls_cv[1]/255:.4f}, S={hls_cv[2]/255:.4f}")
    print(f"       (Raw OpenCV values: {hls_cv})")
    
    ycrcb_cv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[0][0]
    print(f"YCrCb: Y={ycrcb_cv[0]}, Cr={ycrcb_cv[1]}, Cb={ycrcb_cv[2]}")
    print(f"       (Raw OpenCV values: {ycrcb_cv})")
    
    # ============ Comparison Notes ============
    print("\n" + "="*60)
    print("Comparison Notes:")
    print("="*60)
    print("""
1. HSV/HSL Differences:
   - Manual: H in [0, 360]°, S and V/L in [0, 1]
   - OpenCV: H in [0, 180] (divide by 2), S and V/L in [0, 255]
   - OpenCV uses 8-bit representation for efficiency

2. YCrCb Differences:
   - Minor differences may occur due to:
     * Different conversion coefficients
     * Rounding in 8-bit integer arithmetic
     * Implementation-specific optimizations

3. Color Space Usage:
   - HSV/HSL: Good for color-based segmentation
   - YCrCb: Used in JPEG compression, separates luminance from chrominance
""")