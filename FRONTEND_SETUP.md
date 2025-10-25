# AUTHENTICA - Signature Verification System

A professional cyberpunk-themed web application for verifying document signatures using deep learning. Detect forged signatures with high accuracy using a trained Siamese CNN model.

## Features

✨ **Advanced Signature Verification**
- Deep learning-powered Siamese CNN model
- Real-time confidence scoring
- Professional cyberpunk UI with smooth animations

🔐 **Security & Performance**
- GPU acceleration support
- Batch verification capability
- RESTful API design
- CORS-enabled for web integration

📊 **Detailed Analysis**
- Euclidean distance scoring
- Adaptive confidence metrics
- FAR/FRR optimization
- Comprehensive result details

## System Requirements

- **Python**: 3.8+
- **CUDA**: Optional (for GPU acceleration)
- **Memory**: 4GB minimum
- **Browser**: Modern browser (Chrome, Firefox, Safari, Edge)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Snikitha-V/Authentica.git
cd Authentica
```

### 2. Create & Activate Conda Environment

```bash
# Create conda environment
conda create -n authentica-cpu python=3.10 -y
conda activate authentica-cpu

# For GPU support:
# conda create -n authentica-gpu python=3.10 pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch -y
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Model Checkpoint

Ensure `checkpoints/best_model.pth` exists:

```bash
ls -la checkpoints/
# Output should show: best_model.pth
```

If checkpoint is missing, train the model first:

```bash
python train.py --epochs 50 --lr 0.001 --batch-size 32
```

## Running the Application

### Start the Backend API

```bash
# Activate conda environment
conda activate authentica-cpu

# Run Flask server
python app.py
```

Expected output:
```
============================================================
🔐 AUTHENTICA - Signature Verification API
============================================================
Device: cpu
Optimal threshold: 0.573

Starting Flask server on http://localhost:5000
============================================================
```

### Open the Frontend (in another terminal)

```bash
# Option 1: Open in default browser
open index.html

# Option 2: Use Python's simple HTTP server
python3 -m http.server 8000
# Then visit: http://localhost:8000
```

## API Documentation

### Health Check

**Endpoint**: `GET /health`

```bash
curl http://localhost:5000/health
```

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

### Verify Signature

**Endpoint**: `POST /verify`

**Request** (multipart/form-data):
- `real_sig`: Reference signature image
- `suspected_sig`: Suspected signature image

```bash
curl -X POST http://localhost:5000/verify \
  -F "real_sig=@reference.png" \
  -F "suspected_sig=@suspected.png"
```

**Response**:
```json
{
  "success": true,
  "distance": 0.425,
  "threshold": 0.573,
  "confidence": 85.3,
  "is_authentic": true,
  "verdict": "AUTHENTIC",
  "message": "Signature is AUTHENTIC with 85.3% confidence"
}
```

### Batch Verification

**Endpoint**: `POST /batch-verify`

Verify multiple suspected signatures against a single reference.

```bash
curl -X POST http://localhost:5000/batch-verify \
  -F "real_sig=@reference.png" \
  -F "suspected_sigs=@sig1.png" \
  -F "suspected_sigs=@sig2.png"
```

## How to Use the Frontend

1. **Upload Reference Signature**
   - Click the first upload area or drag-and-drop
   - Select the authentic signature image

2. **Upload Suspected Signature**
   - Click the second upload area or drag-and-drop
   - Select the signature to verify

3. **Click "Analyze Signatures"**
   - Wait for the analysis to complete
   - View the detailed results with confidence score

4. **Interpret Results**
   - **Green & Check (✓)**: Signature is AUTHENTIC
   - **Red & X (✗)**: Signature is FORGED
   - **Confidence Score**: 0-100% reliability of verdict

## Confidence Score Interpretation

- **80-100%**: Very high confidence (excellent match)
- **60-79%**: High confidence (good match)
- **40-59%**: Medium confidence (uncertain)
- **0-39%**: Low confidence (likely forged)

## Model Architecture

The system uses a **Siamese Convolutional Neural Network** (Siamese CNN):

- **Input**: Paired signature images (224×224 pixels)
- **Network**: Dual CNN branches with shared weights
- **Output**: Embeddings compared via Euclidean distance
- **Loss**: Contrastive loss function

### Key Metrics

- **Optimal Threshold**: 0.573 (FAR ≈ FRR)
- **Validation AUC**: 0.794
- **Accuracy**: ~71%
- **FAR**: ~30%
- **FRR**: ~29%

## Training

Train the model with custom parameters:

```bash
python train.py --epochs 100 --lr 0.0001 --batch-size 64
```

**Parameters**:
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.0001)
- `--batch-size`: Batch size (default: 32)

Training saves the best model to `checkpoints/best_model.pth`.

## Evaluation

Evaluate the model on test data:

```bash
# With checkpoint
python evaluate.py

# With specific checkpoint
python evaluate.py --checkpoint path/to/checkpoint.pth

# With random model (dry run)
python evaluate.py --allow-random
```

## Troubleshooting

### API Not Starting

**Error**: `Address already in use`
```bash
# Kill process on port 5000
lsof -i :5000
kill -9 <PID>
```

**Error**: `ModuleNotFoundError: No module named 'flask'`
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Model Not Found

**Error**: `FileNotFoundError: No such file or directory: 'checkpoints/best_model.pth'`
```bash
# Train model first
python train.py

# Or download pre-trained checkpoint
# (if available in releases)
```

### Frontend Not Connecting to API

**Error**: `API Offline` badge in UI
```bash
# Ensure Flask server is running
python app.py

# Check CORS headers
curl -i http://localhost:5000/health

# Verify ports
netstat -an | grep 5000
```

### Image Processing Errors

**Supported Formats**: PNG, JPG, JPEG, BMP, GIF
- Maximum file size: 10MB
- Recommended resolution: 200×200 to 500×500 pixels
- Ensure images are clear and well-lit

## Performance Tips

### GPU Acceleration

To enable GPU support:

```bash
# Update PyTorch for CUDA
conda install pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch

# Run app (auto-detects GPU)
python app.py
```

### Batch Processing

For verifying multiple signatures:

```bash
# Use batch endpoint
curl -X POST http://localhost:5000/batch-verify \
  -F "real_sig=@reference.png" \
  -F "suspected_sigs=@batch/*.png"
```

## File Structure

```
Authentica/
├── app.py                 # Flask backend API
├── index.html            # Frontend UI
├── model.py              # Siamese CNN architecture
├── dataset.py            # Dataset handling
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── loss.py               # Contrastive loss
├── config.py             # Configuration
├── utils.py              # Utility functions
├── preprocess.py         # Data preprocessing
├── requirements.txt      # Dependencies
├── README.md             # This file
├── checkpoints/
│   └── best_model.pth   # Trained model
├── data/
│   └── processed/
│       ├── genuine/      # Authentic signatures
│       └── forged/       # Forged signatures
└── uploads/              # Temp uploaded files
```

## Configuration

Edit `config.py` to customize:

```python
# Device (cpu/cuda)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
EMBEDDING_DIM = 128
IMG_SIZE = 224

# Training
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
BATCH_SIZE = 32

# Paths
DATA_DIR_GENUINE = 'data/processed/genuine/'
DATA_DIR_FORGED = 'data/processed/forged/'
CHECKPOINT_DIR = 'checkpoints/'
```

## Deployment

### Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:

```bash
docker build -t authentica .
docker run -p 5000:5000 authentica
```

### Cloud Deployment

- **Heroku**: `Procfile`: `web: gunicorn app:app`
- **AWS**: Deploy to Lambda/EC2 with API Gateway
- **Azure**: App Service with Python runtime
- **GCP**: Cloud Run or App Engine

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Siamese Networks](https://en.wikipedia.org/wiki/Siamese_neural_network)

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributors

- **Snikitha-V** - Original author
- **Akhilesh Kumar Avel** - Frontend & API integration

## Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/Snikitha-V/Authentica/issues)
- Email: support@authentica.dev

---

**Made with ❤️ for document security**
