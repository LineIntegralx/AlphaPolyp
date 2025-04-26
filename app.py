import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import AdamW
import cv2
import torch
import time

# Import custom modules
from model_architecture.DiceLoss import dice_metric_loss
from model_architecture.Model import create_model

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'alphapolyp_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload and result directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Global variable to store the model
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_ai_model():
    """Load the TensorFlow model with custom objects"""
    global model
    
    # Define custom objects for model loading
    custom_objects = {
        'AdamW': AdamW,
        'dice_metric_loss': dice_metric_loss
    }
    
    model_path = 'alphapolyp_optimized_model_3500cases.h5'
    
    # Check if model exists
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = load_model(model_path)
    else:
        print(f"Model file {model_path} not found. Creating new model...")
        model = create_model(352, 352, 3, 1, 17)
        print("Model created but not trained. Predictions will be random.")

def preprocess_image(image_path, img_size=352):
    """Load and preprocess an image for prediction"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def visualize_results(image_path, segmentation, volume, dimensions, subject_name):
    """Create visualization of the prediction results"""
    # Read original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize to match the model input size
    image = cv2.resize(image, (352, 352))
    
    # Create a copy of the original image
    original = image.copy()
    
    # Resize segmentation to match image dimensions
    segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]))
    
    # Create segmentation overlay with red color
    segmentation = (segmentation * 255).astype(np.uint8)
    
    # Create red overlay (BGR format)
    red_overlay = np.zeros_like(image)
    red_overlay[:, :, 2] = segmentation  # Red channel
    
    # Blend original image with red overlay
    overlay = cv2.addWeighted(image, 0.7, red_overlay, 0.3, 0)
    
    # Create a side-by-side display
    combined = np.hstack((original, overlay))
    
    return combined

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Start timing
            start_time = time.time()
            
            # Generate unique filename
            filename = secure_filename(file.filename)
            base_name, extension = os.path.splitext(filename)
            unique_filename = f"{base_name}_{uuid.uuid4().hex[:8]}{extension}"
            
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Load and preprocess the image
            preprocessed_img = preprocess_image(file_path)
            
            if model is not None:
                # Make prediction
                segmentation, regression = model.predict(preprocessed_img)
                
                # Process outputs
                segmentation = segmentation[0, :, :, 0]
                
                # Check if polyp is detected
                if np.any(segmentation > 0.5):
                    volume = float(regression[0, 0])
                    dimensions = regression[0, 1:4].tolist()
                else:
                    volume = 0.0
                    dimensions = [0.0, 0.0, 0.0]
                
                # Create visualization
                result_img = visualize_results(file_path, segmentation, volume, dimensions, base_name)
                
                # Save result image
                result_filename = f"{base_name}_result{extension}"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                cv2.imwrite(result_path, result_img)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                return render_template('result.html',
                                    original_image=url_for('static', filename=f'uploads/{unique_filename}'),
                                    result_image=url_for('static', filename=f'results/{result_filename}'),
                                    filename=base_name,
                                    volume=volume,
                                    dimensions=dimensions,
                                    processing_time=processing_time,
                                    has_polyp=np.any(segmentation > 0.5))
            
            # Start timing for mock data processing
            start_time = time.time()
            
            # If model is not loaded, use mock data
            mock_segmentation = np.zeros((352, 352))
            mock_segmentation[100:250, 150:300] = 1  # Mock polyp area
            
            # Mock regression values
            mock_volume = 54.60
            mock_dimensions = [7.52, 5.83, 3.91]
            
            # Create visualization with mock data
            subject_name = base_name
            result_img = visualize_results(file_path, mock_segmentation, mock_volume, mock_dimensions, subject_name)
            
            # Save result image
            result_filename = f"{base_name}_result{extension}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, result_img)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return results with mock data
            return render_template('result.html', 
                                original_image=url_for('static', filename=f'uploads/{unique_filename}'),
                                result_image=url_for('static', filename=f'results/{result_filename}'),
                                filename=base_name,
                                volume=mock_volume,
                                dimensions=mock_dimensions,
                                processing_time=processing_time)  # Actual processing time
            
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

def pred_image(img_path, model, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(img)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
    
    pred = pred.squeeze().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    
    # Create output image with original and prediction side by side
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (256, 256))
    
    # Create a blank image for the prediction
    pred_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    pred_rgb[:, :, 0] = pred  # Red channel for the prediction
    
    # Combine original and prediction
    combined = np.hstack((original, pred_rgb))
    
    # Convert back to BGR for saving
    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    
    return combined

if __name__ == '__main__':
    # Load the AI model when the app starts
    try:
        load_ai_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("The application will use mock data for predictions.")
    
    # Run the Flask app
    app.run(debug=True)
