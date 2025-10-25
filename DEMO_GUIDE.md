# AUTHENTICA - Getting Started Demo

Welcome to AUTHENTICA! This guide will walk you through a complete workflow to verify signatures using the application.

## 📋 Pre-requisites

- ✓ Flask backend running on `http://localhost:5000`
- ✓ Frontend accessible on `http://localhost:8000`
- ✓ Model checkpoint (`checkpoints/best_model.pth`) present
- ✓ Sample signature images for testing

## 🎬 Step-by-Step Demo

### Step 1: Verify System Health

Before starting verification, ensure the API is online:

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

The frontend will automatically show this as a green `● API Online` badge.

### Step 2: Prepare Test Signatures

You'll need two signature images:

1. **Reference Signature** (Authentic sample)
   - A clean, clear authentic signature
   - Format: PNG or JPG
   - Size: 200×200 to 500×500 pixels recommended

2. **Suspected Signature** (To verify)
   - A signature that may or may not be forged
   - Same format and quality requirements
   - Can be slightly different style to test flexibility

### Step 3: Upload Reference Signature

1. Open **index.html** in your browser
2. You'll see the cyberpunk-themed interface with two upload areas:
   ```
   ┌─────────────────────────────────────┐
   │  ◆ AUTHENTICA ◆                     │
   │  Advanced Signature Verification    │
   └─────────────────────────────────────┘
   
   ┌─────────────────┐    ┌──────────────┐
   │ Reference Sig   │    │ Verification │
   │ Upload Area     │    │ Result       │
   └─────────────────┘    └──────────────┘
   ```

3. Click on the **"Reference Signature"** upload area (top left)
4. Select your authentic signature image
5. Confirm it shows: `✓ filename.png`

### Step 4: Upload Suspected Signature

1. Click on the **"Suspected Signature"** upload area (bottom left)
2. Select the signature to verify
3. Confirm it shows: `✓ filename.png`

### Step 5: Analyze Signatures

1. Click the **"ANALYZE SIGNATURES"** button
2. You'll see a loading spinner: `⏳ Analyzing signatures...`
3. Wait for processing (usually 1-3 seconds on CPU)

### Step 6: Review Results

The right panel will display results with:

#### A) Verdict Display
```
┌──────────────────────────────┐
│         ✓ or ✗               │ (Icon changes)
│      AUTHENTIC               │ (or FORGED)
└──────────────────────────────┘
```

#### B) Confidence Score
Visual progress bar showing:
- **80-100%**: Very High Confidence
- **60-79%**: High Confidence  
- **40-59%**: Medium Confidence
- **0-39%**: Low Confidence

#### C) Detailed Metrics
| Metric | Example |
|--------|---------|
| Status | AUTHENTIC |
| Distance Score | 0.4250 |
| Threshold | 0.5730 |
| Message | Signature is AUTHENTIC with 85.3% confidence |

## 🎨 Interface Guide

### Color Meanings (Cyberpunk Theme)

| Color | Meaning |
|-------|---------|
| **Cyan (0ff)** | Primary information, highlights |
| **Magenta (f0f)** | Secondary accents, hover states |
| **Green (success)** | Authentic verdict, positive feedback |
| **Red (danger)** | Forged verdict, warnings |
| **Yellow (warning)** | Upload indicators, cautions |

### Interactive Elements

- **Upload Areas**: Glow on hover, become highlighted when files are dropped
- **File Badges**: Show selected files with quick remove button (✕)
- **Analyze Button**: Becomes enabled after both images uploaded
- **Loading Animation**: Spinning circle during processing
- **Confidence Bar**: Fills from left to right with animated gradient

## 📊 Interpreting Results

### Example 1: Authentic Match
```
✓ AUTHENTIC
█████████░ 92.3%

Distance: 0.3120
Threshold: 0.5730
Status: AUTHENTIC
Message: Signature is AUTHENTIC with 92.3% confidence
```
**Interpretation**: Excellent match! This is almost certainly the same person.

### Example 2: Questionable Match
```
✓ AUTHENTIC
██████░░░ 61.5%

Distance: 0.4920
Threshold: 0.5730
Status: AUTHENTIC
Message: Signature is AUTHENTIC with 61.5% confidence
```
**Interpretation**: Likely authentic but recommend manual verification.

### Example 3: Likely Forged
```
✗ FORGED
███░░░░░░ 38.2%

Distance: 0.7410
Threshold: 0.5730
Status: FORGED
Message: Signature is FORGED with high confidence
```
**Interpretation**: Distance exceeds threshold. Likely a forgery.

## 🔄 Multiple Verifications

You can verify as many signatures as you need against the reference:

1. Keep the reference signature
2. Click **"Remove"** (✕) on the suspected signature
3. Upload a new suspected signature
4. Click **"ANALYZE SIGNATURES"** again
5. Results update automatically

## 💾 Distance Score Explanation

The system calculates **Euclidean distance** between signature embeddings:

- **Distance < 0.573**: Model predicts AUTHENTIC
- **Distance ≥ 0.573**: Model predicts FORGED

### Distance Scale
```
0.0 ──────────────────────── 1.0 ──────────────────────── 2.0
Perfect Match          Uncertain          Completely Different
```

## 🚨 Error Handling

### "API Offline"
- Backend server not running
- **Fix**: Run `python app.py` in terminal

### "Image preprocessing failed"
- Invalid image format or corruption
- **Fix**: Try a different image, ensure PNG/JPG format

### "Empty file uploaded"
- File is corrupted or empty
- **Fix**: Re-select a valid image

### "File exceeds size limit"
- Image file > 10MB
- **Fix**: Compress image before uploading

## 🎯 Best Practices

### For Accurate Results

1. **Use High-Quality Images**
   - Clear, well-lit photos
   - Avoid blurry or distorted scans
   - Consistent lighting

2. **Signature Positioning**
   - Similar orientation and scaling
   - Fill about 60-80% of image
   - Minimal background

3. **Multiple Samples**
   - Compare against multiple authentic samples
   - Natural variations occur in signatures
   - Use average of multiple verifications

4. **Manual Verification**
   - Always have human review for high-stakes docs
   - Use confidence score as guidance, not absolute truth
   - Follow legal/compliance requirements

## 📈 Advanced: Batch Verification

For verifying multiple documents, use the API directly:

```bash
curl -X POST http://localhost:5000/batch-verify \
  -F "real_sig=@authentic.png" \
  -F "suspected_sigs=@doc1.png" \
  -F "suspected_sigs=@doc2.png" \
  -F "suspected_sigs=@doc3.png"
```

Response:
```json
{
  "success": true,
  "total": 3,
  "results": [
    {"filename": "doc1.png", "verdict": "AUTHENTIC", "confidence": 87.5},
    {"filename": "doc2.png", "verdict": "FORGED", "confidence": 42.1},
    {"filename": "doc3.png", "verdict": "AUTHENTIC", "confidence": 79.3}
  ]
}
```

## 🔐 Privacy & Security

- **No Cloud Upload**: All processing happens locally
- **No Storage**: Uploaded files are not saved
- **No Tracking**: No analytics or logging of signatures
- **Model Only**: Checkpoint file is the only stored asset

## 📞 Troubleshooting

### System Not Responding
```bash
# Check if backend is running
curl -v http://localhost:5000/health

# Check if frontend is accessible
curl -v http://localhost:8000
```

### Clear Cache
```bash
# Browser cache
Ctrl+Shift+R (or Cmd+Shift+R on Mac)

# Local storage
Open DevTools (F12) → Application → Clear All
```

### View Logs
```bash
# Backend logs
tail -f /tmp/authentica_backend.log

# Frontend errors
Open DevTools (F12) → Console
```

## 🎓 Learning More

- **Model Architecture**: See `model.py` for Siamese CNN details
- **Training Process**: See `train.py` for model training
- **API Documentation**: See `FRONTEND_SETUP.md`
- **Integration Guide**: See `INTEGRATION_GUIDE.md`

## ✨ Pro Tips

1. **Faster Processing**: Use GPU if available
   ```bash
   # Install GPU support
   conda install pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch
   ```

2. **Batch Automation**: Create script to process multiple docs
   ```python
   # See FRONTEND_SETUP.md for Python client example
   ```

3. **Threshold Tuning**: Adjust sensitivity based on use case
   - Edit `app.py`: `optimal_threshold = 0.573`

4. **Custom Styling**: Modify CSS in `index.html`
   - Change colors: Update `:root` CSS variables
   - Animations: Edit `@keyframes` sections

---

## 🎉 You're Ready!

Your signature verification system is fully operational. Start by verifying some test signatures and exploring the interface!

**Questions?** Check the troubleshooting section or review the detailed documentation in the project.

---

**Happy Verifying! 🔐✨**
