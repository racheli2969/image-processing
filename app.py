"""
Flask Web Application for Image Processing
Allows users to upload images and apply various processing techniques
"""

from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import os
import io
import base64
from datetime import datetime

# Import helper functions from the 1 folder
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '1'))
from helper_functions import compute_histogram, histogram_stretching, stretch_histogram

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    """Generate a unique filename with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    name, ext = os.path.splitext(secure_filename(original_filename))
    return f"{name}_{timestamp}{ext}"

def process_rgb_channels(img_path):
    """Process: Display RGB channels separately"""
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    
    # Split channels
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Red channel
    axes[0, 1].imshow(r_channel, cmap='Reds')
    axes[0, 1].set_title('Red Channel', fontsize=14)
    axes[0, 1].axis('off')
    
    # Green channel
    axes[1, 0].imshow(g_channel, cmap='Greens')
    axes[1, 0].set_title('Green Channel', fontsize=14)
    axes[1, 0].axis('off')
    
    # Blue channel
    axes[1, 1].imshow(b_channel, cmap='Blues')
    axes[1, 1].set_title('Blue Channel', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save to file
    output_filename = generate_unique_filename('rgb_channels.png')
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_filename

def process_grayscale_stretch(img_path):
    """Process: Grayscale with histogram stretching"""
    img = Image.open(img_path).convert('RGB')
    
    # Convert to grayscale
    gray_img = img.convert('L')
    gray_array = np.array(gray_img)
    
    # Apply histogram stretching
    stretched_array = histogram_stretching(gray_array)
    stretched_img = Image.fromarray(stretched_array, 'L')
    
    # Compute histograms
    original_hist = compute_histogram(gray_array)
    stretched_hist = compute_histogram(stretched_array)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original grayscale
    axes[0, 0].imshow(gray_img, cmap='gray')
    axes[0, 0].set_title('Original Grayscale', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Original histogram
    axes[0, 1].bar(range(256), original_hist, color='gray', width=1)
    axes[0, 1].set_title('Original Histogram')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xlim([0, 255])
    
    # Original stats
    axes[0, 2].text(0.5, 0.5, f'Min: {gray_array.min()}\nMax: {gray_array.max()}\nMean: {gray_array.mean():.1f}',
                    ha='center', va='center', fontsize=16, transform=axes[0, 2].transAxes)
    axes[0, 2].set_title('Original Stats')
    axes[0, 2].axis('off')
    
    # Stretched grayscale
    axes[1, 0].imshow(stretched_img, cmap='gray')
    axes[1, 0].set_title('Histogram Stretched', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Stretched histogram
    axes[1, 1].bar(range(256), stretched_hist, color='darkblue', width=1)
    axes[1, 1].set_title('Stretched Histogram')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xlim([0, 255])
    
    # Stretched stats
    axes[1, 2].text(0.5, 0.5, f'Min: {stretched_array.min()}\nMax: {stretched_array.max()}\nMean: {stretched_array.mean():.1f}',
                    ha='center', va='center', fontsize=16, transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Stretched Stats')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save to file
    output_filename = generate_unique_filename('grayscale_stretch.png')
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_filename

def process_color_stretch(img_path):
    """Process: Color histogram stretching (each channel separately)"""
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    
    # Split and stretch each channel
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]
    
    r_stretched = histogram_stretching(r_channel)
    g_stretched = histogram_stretching(g_channel)
    b_stretched = histogram_stretching(b_channel)
    
    # Combine stretched channels
    stretched_array = np.stack([r_stretched, g_stretched, b_stretched], axis=2)
    stretched_img = Image.fromarray(stretched_array.astype(np.uint8))
    
    # Compute histograms
    r_hist = compute_histogram(r_channel)
    g_hist = compute_histogram(g_channel)
    b_hist = compute_histogram(b_channel)
    
    r_hist_stretched = compute_histogram(r_stretched)
    g_hist_stretched = compute_histogram(g_stretched)
    b_hist_stretched = compute_histogram(b_stretched)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Original histograms
    axes[0, 1].bar(range(256), r_hist, color='red', alpha=0.6, width=1)
    axes[0, 1].set_title('Red Channel (Original)')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[0, 2].bar(range(256), g_hist, color='green', alpha=0.6, width=1)
    axes[0, 2].set_title('Green Channel (Original)')
    axes[0, 2].set_xlabel('Pixel Value')
    axes[0, 2].set_ylabel('Frequency')
    
    axes[0, 3].bar(range(256), b_hist, color='blue', alpha=0.6, width=1)
    axes[0, 3].set_title('Blue Channel (Original)')
    axes[0, 3].set_xlabel('Pixel Value')
    axes[0, 3].set_ylabel('Frequency')
    
    # Stretched image
    axes[1, 0].imshow(stretched_img)
    axes[1, 0].set_title('Histogram Stretched', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Stretched histograms
    axes[1, 1].bar(range(256), r_hist_stretched, color='red', alpha=0.6, width=1)
    axes[1, 1].set_title('Red Channel (Stretched)')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    axes[1, 2].bar(range(256), g_hist_stretched, color='green', alpha=0.6, width=1)
    axes[1, 2].set_title('Green Channel (Stretched)')
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Frequency')
    
    axes[1, 3].bar(range(256), b_hist_stretched, color='blue', alpha=0.6, width=1)
    axes[1, 3].set_title('Blue Channel (Stretched)')
    axes[1, 3].set_xlabel('Pixel Value')
    axes[1, 3].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save to file
    output_filename = generate_unique_filename('color_stretch.png')
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_filename

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Handle image upload and processing"""
    if 'image' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['image']
    processing_type = request.form.get('processing_type', 'rgb_channels')
    
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    if not allowed_file(file.filename):
        return render_template('index.html', error='Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP')
    
    try:
        # Save uploaded file
        filename = generate_unique_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process based on selected type
        if processing_type == 'rgb_channels':
            result_filename = process_rgb_channels(filepath)
            description = 'RGB Color Channels Separation'
        elif processing_type == 'grayscale_stretch':
            result_filename = process_grayscale_stretch(filepath)
            description = 'Grayscale with Histogram Stretching'
        elif processing_type == 'color_stretch':
            result_filename = process_color_stretch(filepath)
            description = 'Color Histogram Stretching'
        else:
            return render_template('index.html', error='Invalid processing type')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('result.html', 
                             result_image=result_filename,
                             description=description,
                             processing_type=processing_type)
    
    except Exception as e:
        return render_template('index.html', error=f'Processing error: {str(e)}')

@app.route('/download/<filename>')
def download(filename):
    """Download processed image"""
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
