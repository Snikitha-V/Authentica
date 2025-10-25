# AUTHENTICA - Frontend & Backend Integration Guide

## 🚀 Quick Start (All Platforms)

### Option 1: Automated Start (Recommended)

**macOS / Linux:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
```cmd
start.bat
```

This will:
- ✓ Setup conda environment
- ✓ Install dependencies
- ✓ Start Flask backend on `http://localhost:5000`
- ✓ Start web server on `http://localhost:8000`
- ✓ Open frontend in browser automatically

### Option 2: Manual Start

#### Terminal 1: Start Backend
```bash
conda activate authentica-cpu
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

#### Terminal 2: Start Frontend
```bash
# Option A: Python HTTP Server
python3 -m http.server 8000

# Option B: Node.js (if installed)
npx serve -l 8000

# Option C: Open directly in browser
open index.html
```

Then visit: **http://localhost:8000**

## 📁 Project Structure

```
Authentica/
│
├── Backend (Flask API)
│   ├── app.py                 # Main Flask server
│   ├── model.py               # Siamese CNN architecture
│   ├── dataset.py             # Dataset utilities
│   ├── loss.py                # Contrastive loss
│   ├── config.py              # Configuration
│   ├── utils.py               # Helper functions
│   └── checkpoints/
│       └── best_model.pth     # Trained model
│
├── Frontend (Web UI)
│   ├── index.html             # Main application
│   ├── FRONTEND_SETUP.md      # Detailed docs
│   └── start.sh / start.bat   # Quick launchers
│
├── Training & Evaluation
│   ├── train.py               # Model training
│   ├── evaluate.py            # Model evaluation
│   └── preprocess.py          # Data preprocessing
│
└── Dependencies
    └── requirements.txt       # Python packages
```

## 🔧 API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### Verify Signature
```bash
curl -X POST http://localhost:5000/verify \
  -F "real_sig=@reference_signature.png" \
  -F "suspected_sig=@suspected_signature.png"
```

Response:
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
```bash
curl -X POST http://localhost:5000/batch-verify \
  -F "real_sig=@reference.png" \
  -F "suspected_sigs=@sig1.png" \
  -F "suspected_sigs=@sig2.png"
```

## 🎨 Frontend Features

### User Interface
- **Cyberpunk Theme**: Neon cyan & magenta with animated grid
- **Drag & Drop**: Upload signatures by dragging
- **Real-time Feedback**: File validation and upload status
- **Confidence Visualization**: Animated progress bar
- **Detailed Results**: Distance, threshold, verdict

### Supported Formats
- PNG, JPG, JPEG, BMP, GIF
- Maximum size: 10MB
- Recommended: 200×200 to 500×500 pixels

### Result Interpretation

| Verdict | Confidence | Meaning |
|---------|-----------|---------|
| ✓ AUTHENTIC | 80-100% | Excellent match, highly trusted |
| ✓ AUTHENTIC | 60-79% | Good match, trusted |
| ✓ AUTHENTIC | 40-59% | Medium confidence, verify manually |
| ✗ FORGED | 40-59% | Likely forged, verify manually |
| ✗ FORGED | 60-79% | Likely forged, high confidence |
| ✗ FORGED | 80-100% | Definite forgery, very high confidence |

## 🐛 Troubleshooting

### Backend Won't Start

**Error: `Address already in use`**
```bash
# Kill existing process
lsof -i :5000 | grep -v COMMAND | awk '{print $2}' | xargs kill -9
# Retry
python app.py
```

**Error: `No module named 'flask'`**
```bash
conda activate authentica-cpu
pip install Flask Flask-CORS
```

**Error: `ModuleNotFoundError: model`**
```bash
# Ensure you're in the Authentica directory
cd /path/to/Authentica
python app.py
```

### Frontend Not Connecting

**Issue: "API Offline" badge**

1. Check backend is running:
```bash
curl http://localhost:5000/health
# Should return: {"status": "ok", "model_loaded": true, ...}
```

2. Check CORS headers:
```bash
curl -i http://localhost:5000/health
# Should include: Access-Control-Allow-Origin: *
```

3. Verify ports aren't in use:
```bash
# macOS/Linux
lsof -i :5000
lsof -i :8000

# Windows
netstat -ano | findstr :5000
netstat -ano | findstr :8000
```

### Image Upload Issues

**Error: "Image preprocessing failed"**
- Check image format (PNG/JPG recommended)
- Ensure file size < 10MB
- Try converting image: `convert input.png -resize 300x300 input_resized.png`

**Error: "File exceeds size limit"**
- Compress image: `convert large.png -quality 85 small.png`
- Or use ImageMagick: `mogrify -resize 50% image.png`

## 🚀 Advanced Configuration

### Custom Port
Edit `app.py`:
```python
app.run(host='0.0.0.0', port=5001)  # Change 5001
```

### Enable Debug Mode
```bash
# Automatically reloads on code changes
python app.py --debug
```

### Use GPU
Ensure CUDA-enabled PyTorch is installed:
```bash
conda install pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch
# Run app (auto-detects GPU)
python app.py
```

## 📊 Performance Metrics

### Model Performance
- **Accuracy**: ~71%
- **AUC**: 0.794
- **FAR (False Acceptance Rate)**: ~30%
- **FRR (False Rejection Rate)**: ~29%
- **Optimal Threshold**: 0.573

### Processing Speed
- **GPU**: ~50ms per comparison
- **CPU**: ~200ms per comparison
- **Batch Processing**: Efficient for 10+ signatures

## 🔐 Security Considerations

1. **Input Validation**: All uploads validated for type & size
2. **CORS Protection**: Frontend only accepts same-origin requests
3. **No Persistence**: Uploaded files not stored permanently
4. **Model Protection**: Checkpoint file backed up

## 📝 Example Usage

### Python Client
```python
import requests
from pathlib import Path

BASE_URL = "http://localhost:5000"

# Verify single signature
with open("reference.png", "rb") as ref, \
     open("suspected.png", "rb") as susp:
    files = {
        'real_sig': ref,
        'suspected_sig': susp
    }
    response = requests.post(f"{BASE_URL}/verify", files=files)
    result = response.json()
    
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"Message: {result['message']}")
```

### JavaScript Client
```javascript
async function verifySignature(realFile, suspectedFile) {
    const formData = new FormData();
    formData.append('real_sig', realFile);
    formData.append('suspected_sig', suspectedFile);
    
    const response = await fetch('http://localhost:5000/verify', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    console.log(`Verdict: ${result.verdict}`);
    console.log(`Confidence: ${result.confidence}%`);
    return result;
}
```

## 📚 Additional Resources

- **Training**: `python train.py --help`
- **Evaluation**: `python evaluate.py --help`
- **Model Architecture**: See `model.py`
- **Dataset Format**: See `dataset.py`

## 🎯 Next Steps

1. ✅ Backend running on port 5000
2. ✅ Frontend running on port 8000
3. ✅ Test with sample signatures
4. 📝 Train on your dataset
5. 🚀 Deploy to production

## 📞 Support

- **Issues**: Check terminal output for error logs
- **Backend Log**: `/tmp/authentica_backend.log`
- **Frontend Log**: `/tmp/authentica_frontend.log`
- **Debug**: Run with verbose output

---

**Your signature verification system is ready! 🔐**
