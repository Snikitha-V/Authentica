# 🔐 AUTHENTICA - Signature Verification System

> **Professional Cyberpunk-Themed Deep Learning Application for Document Signature Verification**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## ✨ Features

### 🎯 Core Capabilities
- **Siamese CNN Model**: Advanced deep learning architecture for signature comparison
- **Real-time Verification**: Instant analysis with confidence scoring
- **Professional Dashboard**: Cyberpunk-themed UI with smooth animations
- **RESTful API**: Easy integration with external systems
- **Batch Processing**: Verify multiple signatures efficiently

### 🎨 User Experience
- **Drag & Drop Upload**: Intuitive file handling
- **Live Feedback**: Real-time validation and status updates
- **Confidence Visualization**: Animated progress indicators
- **Detailed Metrics**: Distance scores, thresholds, verdicts
- **Responsive Design**: Works on desktop and tablets

### 🚀 Technical Excellence
- **GPU Acceleration**: CUDA support for faster processing
- **Production Ready**: Error handling, CORS, input validation
- **Modular Architecture**: Easy to extend and customize
- **Comprehensive Logging**: Debug-friendly error messages

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 70.68% |
| AUC Score | 0.794 |
| FAR (False Acceptance) | 29.65% |
| FRR (False Rejection) | 29.02% |
| Optimal Threshold | 0.573 |

## 🚀 Quick Start

### One-Command Setup (Recommended)

**macOS / Linux:**
```bash
chmod +x start.sh && ./start.sh
```

**Windows:**
```cmd
start.bat
```

This automatically:
- ✅ Creates conda environment
- ✅ Installs dependencies
- ✅ Starts Flask backend (port 5000)
- ✅ Starts web server (port 8000)
- ✅ Opens frontend in browser

### Manual Setup

**1. Clone Repository**
```bash
git clone https://github.com/Snikitha-V/Authentica.git
cd Authentica
```

**2. Create Environment**
```bash
conda create -n authentica-cpu python=3.10 -y
conda activate authentica-cpu
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
pip install Flask Flask-CORS  # Web framework
```

**4. Start Services**

Terminal 1 - Backend:
```bash
conda activate authentica-cpu
python app.py
```

Terminal 2 - Frontend:
```bash
python3 -m http.server 8000
# Then open: http://localhost:8000
```

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| [FRONTEND_SETUP.md](FRONTEND_SETUP.md) | Complete setup guide & API documentation |
| [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) | Frontend/backend integration details |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | Step-by-step usage tutorial |

## 🎯 How to Use

### Via Web Interface

1. **Open Dashboard**
   ```
   http://localhost:8000
   ```

2. **Upload Reference Signature**
   - Click "Reference Signature" upload area
   - Select authentic signature image
   - File appears with ✓ checkmark

3. **Upload Suspected Signature**
   - Click "Suspected Signature" upload area
   - Select signature to verify
   - File appears with ✓ checkmark

4. **Click "ANALYZE SIGNATURES"**
   - Loading spinner appears
   - Analysis completes in 1-3 seconds

5. **Review Results**
   - **Green ✓**: Signature is AUTHENTIC
   - **Red ✗**: Signature is FORGED
   - Confidence score 0-100%
   - Detailed metrics

### Via REST API

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Verify Signature:**
```bash
curl -X POST http://localhost:5000/verify \
  -F "real_sig=@reference.png" \
  -F "suspected_sig=@suspected.png"
```

**Response:**
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

## 🏗️ Project Structure

```
Authentica/
├── 🌐 Frontend
│   ├── index.html           # Main web interface
│   ├── start.sh            # macOS/Linux launcher
│   └── start.bat           # Windows launcher
│
├── 🔧 Backend (Flask API)
│   ├── app.py              # Main Flask server
│   ├── model.py            # Siamese CNN
│   ├── dataset.py          # Data utilities
│   ├── loss.py             # Contrastive loss
│   ├── config.py           # Settings
│   └── utils.py            # Helpers
│
├── 📚 Training & Evaluation
│   ├── train.py            # Model training
│   ├── evaluate.py         # Model evaluation
│   └── preprocess.py       # Data processing
│
├── 🎓 Documentation
│   ├── README.md                # This file
│   ├── FRONTEND_SETUP.md        # Setup guide
│   ├── INTEGRATION_GUIDE.md     # Integration details
│   └── DEMO_GUIDE.md            # Usage tutorial
│
├── 🤖 Model & Data
│   ├── checkpoints/
│   │   └── best_model.pth      # Trained Siamese CNN
│   ├── data/processed/
│   │   ├── genuine/            # Authentic signatures
│   │   └── forged/             # Forged signatures
│   └── uploads/                # Temporary files
│
└── 📦 Dependencies
    └── requirements.txt        # Python packages
```

## 🎨 UI Design

### Color Scheme (Cyberpunk Theme)
```
Primary:    Cyan (0ff) - Information & highlights
Secondary:  Magenta (f0f) - Accents & hover states
Success:    Green (0f0) - Authentic verdict
Danger:     Red (f05) - Forged verdict
Background: Dark Blue - Professional, minimal
```

### Key Features
- ✨ Animated grid background
- 🌟 Glowing text effects
- 🎬 Smooth transitions
- 📊 Animated progress bars
- 🎯 Interactive drag-drop zones

## 🔬 Model Architecture

### Siamese CNN
```
Input: Two signature images (224×224)
           ↓
    [Shared CNN Backbone]
           ↓
    [Embedding Layer: 128D]
           ↓
    [Euclidean Distance]
           ↓
Output: Distance score
(< 0.573 = Authentic, ≥ 0.573 = Forged)
```

### Training Details
- **Architecture**: ResNet-like CNN with shared weights
- **Loss Function**: Contrastive loss
- **Optimizer**: Adam (lr=0.0001)
- **Dataset**: Signature image pairs
- **Hardware**: CPU/GPU compatible

## 📈 Model Training

### Train from Scratch
```bash
python train.py \
  --epochs 100 \
  --lr 0.0001 \
  --batch-size 32
```

### Training Parameters
| Parameter | Default | Range |
|-----------|---------|-------|
| epochs | 50 | 10-200 |
| lr | 0.0001 | 0.00001-0.01 |
| batch-size | 32 | 16-128 |

### Training Output
Saves best model (by AUC) to `checkpoints/best_model.pth`

## 🧪 Model Evaluation

### Evaluate on Test Set
```bash
# With checkpoint
python evaluate.py

# With specific checkpoint
python evaluate.py --checkpoint path/to/checkpoint.pth

# With random model (dry run)
python evaluate.py --allow-random
```

### Output
```
Optimal threshold (FAR ≈ FRR): 0.573

Evaluation metrics at optimal threshold:
accuracy: 0.7068
AUC: 0.7942
FAR: 0.2965
FRR: 0.2902
```

## 🐛 Troubleshooting

### Backend Issues

**Port 5000 in use:**
```bash
lsof -i :5000 | grep -v COMMAND | awk '{print $2}' | xargs kill -9
```

**Flask not installed:**
```bash
conda activate authentica-cpu
pip install Flask Flask-CORS
```

**Model not found:**
```bash
# Train model
python train.py

# Or download from releases
# (if available)
```

### Frontend Issues

**API offline badge:**
- Verify backend running: `curl http://localhost:5000/health`
- Check CORS headers are present
- Verify ports 5000 & 8000 are open

**Image upload fails:**
- Use PNG/JPG format
- Ensure file < 10MB
- Try different image file

**Clear browser cache:**
```bash
# macOS/Linux
Cmd+Shift+R

# Windows/Linux
Ctrl+Shift+R
```

## 🔐 Security

- ✅ Input validation for all uploads
- ✅ File size limits (10MB max)
- ✅ CORS protection
- ✅ No persistent storage
- ✅ Local processing only

## 📦 Dependencies

```
torch>=2.0           # Deep learning framework
torchvision>=0.15    # Image utilities
Flask>=2.0           # Web framework
Flask-CORS>=3.0      # CORS support
scikit-learn>=1.0    # ML utilities
Pillow>=9.0          # Image processing
numpy>=1.20          # Numerical computing
```

## 🚀 Deployment

### Docker
```bash
docker build -t authentica .
docker run -p 5000:5000 authentica
```

### Cloud Platforms
- **Heroku**: Deploy Flask with Procfile
- **AWS**: Lambda + API Gateway
- **Azure**: App Service
- **GCP**: Cloud Run

## 📞 Support & Contributing

**Issues?**
- Check [DEMO_GUIDE.md](DEMO_GUIDE.md) troubleshooting
- Review error messages in terminal
- Open GitHub issue with details

**Want to contribute?**
- Fork repository
- Create feature branch
- Submit pull request

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 👥 Credits

- **Original Author**: Snikitha-V
- **Frontend & API**: Akhilesh Kumar Avel
- **Deep Learning**: PyTorch Team

## 🎓 References

- [Siamese Neural Networks for One-shot Learning](https://arxiv.org/abs/1503.03832)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

<div align="center">

### 🎉 Ready to Verify Signatures?

**[Quick Start Guide](DEMO_GUIDE.md)** • **[Full Documentation](FRONTEND_SETUP.md)** • **[API Reference](INTEGRATION_GUIDE.md)**

**Made with ❤️ for document security**

</div>
