"""
Flask backend API for signature verification using trained Siamese CNN model.
Provides endpoints for health check and signature comparison.
"""

import os
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from dataset import get_transforms
from model import SiameseCNN
from config import DEVICE, CHECKPOINT_DIR, EMBEDDING_DIM, IMG_SIZE
import utils

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model state
model = None
eval_transform = None
device = DEVICE
optimal_threshold = 0.573  # From evaluate.py output

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained Siamese CNN model."""
    global model, eval_transform
    
    model = SiameseCNN(EMBEDDING_DIM).to(device)
    eval_transform = get_transforms(IMG_SIZE, train=False)
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
            model_state = model.state_dict()
            filtered_state = {k: v for k, v in state.items() 
                            if k in model_state and model_state[k].shape == v.shape}
            model.load_state_dict(filtered_state, strict=False)
            print(f"✓ Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            raise
    else:
        print(f"✗ Checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    model.eval()

def preprocess_image(image_bytes):
    """Convert image bytes to tensor (converts to grayscale for model)."""
    try:
        # Convert to grayscale (L) to match training data format
        img = Image.open(BytesIO(image_bytes)).convert('L')
        if eval_transform:
            tensor = eval_transform(img)
        else:
            raise RuntimeError("Transforms not initialized")
        return tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def compute_confidence_score(distance, threshold=None):
    """
    Convert Euclidean distance to confidence score.
    Returns (confidence, is_authentic) where:
    - confidence: 0-100 (higher = more confident it's authentic)
    - is_authentic: True if distance < threshold
    """
    if threshold is None:
        threshold = optimal_threshold
    
    # Distance-based confidence
    # If distance is very small -> high confidence it's authentic
    # If distance is very large -> high confidence it's forged
    max_distance = 2.0  # empirical max distance observed
    normalized_distance = min(distance / max_distance, 1.0)
    
    is_authentic = distance < threshold
    
    if is_authentic:
        # Confidence increases as distance decreases below threshold
        confidence = (1.0 - (distance / threshold)) * 100
    else:
        # Confidence increases as distance increases above threshold
        confidence = ((distance - threshold) / (max_distance - threshold)) * 100
    
    confidence = np.clip(confidence, 0, 100)
    return float(confidence), bool(is_authentic)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device)
    }), 200

@app.route('/verify', methods=['POST'])
def verify_signature():
    """
    Verify if a suspected signature matches a real signature.
    Expects multipart form data with:
    - real_sig: reference signature image
    - suspected_sig: signature to verify
    """
    try:
        # Validate request
        if 'real_sig' not in request.files or 'suspected_sig' not in request.files:
            return jsonify({
                'error': 'Missing required files: real_sig and suspected_sig'
            }), 400
        
        real_file = request.files['real_sig']
        suspected_file = request.files['suspected_sig']
        
        if real_file.filename == '' or suspected_file.filename == '':
            return jsonify({'error': 'No selected files'}), 400
        
        if not (allowed_file(real_file.filename) and allowed_file(suspected_file.filename)):
            return jsonify({'error': f'Allowed file types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Read and preprocess images
        real_bytes = real_file.read()
        suspected_bytes = suspected_file.read()
        
        if len(real_bytes) == 0 or len(suspected_bytes) == 0:
            return jsonify({'error': 'Empty file uploaded'}), 400
        
        real_img_tensor = preprocess_image(real_bytes).to(device)
        suspected_img_tensor = preprocess_image(suspected_bytes).to(device)
        
        # Forward pass through model
        with torch.no_grad():
            embedding_real, embedding_suspected = model(real_img_tensor, suspected_img_tensor)
            distance = F.pairwise_distance(embedding_real, embedding_suspected).item()
        
        # Compute confidence
        confidence, is_authentic = compute_confidence_score(distance)
        
        return jsonify({
            'success': True,
            'distance': float(distance),
            'threshold': float(optimal_threshold),
            'confidence': confidence,
            'is_authentic': is_authentic,
            'verdict': 'AUTHENTIC' if is_authentic else 'FORGED',
            'message': f"Signature is {('AUTHENTIC' if is_authentic else 'FORGED')} with {confidence:.1f}% confidence"
        }), 200
    
    except ValueError as e:
        return jsonify({'error': f'Image processing error: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Verification failed: {str(e)}'}), 500

@app.route('/batch-verify', methods=['POST'])
def batch_verify():
    """
    Verify multiple suspected signatures against a reference.
    Expects JSON with:
    - real_sig_path: path to reference signature
    - suspected_files: list of file paths to verify
    """
    try:
        data = request.get_json()
        
        if not data or 'real_sig' not in request.files or 'suspected_sigs' not in request.files:
            return jsonify({'error': 'Invalid request format'}), 400
        
        real_file = request.files['real_sig']
        suspected_files = request.files.getlist('suspected_sigs')
        
        if not suspected_files:
            return jsonify({'error': 'No suspected signatures provided'}), 400
        
        real_bytes = real_file.read()
        real_img_tensor = preprocess_image(real_bytes).to(device)
        
        results = []
        for suspected_file in suspected_files:
            if allowed_file(suspected_file.filename):
                try:
                    suspected_bytes = suspected_file.read()
                    suspected_img_tensor = preprocess_image(suspected_bytes).to(device)
                    
                    with torch.no_grad():
                        embedding_real, embedding_suspected = model(real_img_tensor, suspected_img_tensor)
                        distance = F.pairwise_distance(embedding_real, embedding_suspected).item()
                    
                    confidence, is_authentic = compute_confidence_score(distance)
                    results.append({
                        'filename': suspected_file.filename,
                        'distance': float(distance),
                        'confidence': confidence,
                        'verdict': 'AUTHENTIC' if is_authentic else 'FORGED'
                    })
                except Exception as e:
                    results.append({
                        'filename': suspected_file.filename,
                        'error': str(e)
                    })
        
        return jsonify({
            'success': True,
            'total': len(suspected_files),
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Batch verification failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("\n" + "="*60)
    print("🔐 AUTHENTICA - Signature Verification API")
    print("="*60)
    print(f"Device: {device}")
    print(f"Optimal threshold: {optimal_threshold}")
    print("\nStarting Flask server on http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
