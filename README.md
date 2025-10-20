# Authentica

Authentica is a production-oriented Siamese neural network for offline signature verification. The system ingests paired genuine and forged signatures, learns discriminative embeddings with a ResNet18 backbone, and evaluates pairwise similarity using a contrastive objective. This README documents the complete workflow, environment, configuration, and results so you can reproduce or extend the project confidently.

---

## Performance/Results 

| Split | Accuracy | AUC | FAR | FRR | Optimal Threshold |
|-------|----------|-----|-----|-----|-------------------|
| Validation | 0.9455 | 0.9893 | 0.0513 | 0.0575 | 0.021 |

- Metrics generated with `python evaluate.py` after running `python train.py --epochs 40 --batch-size 16` in the CUDA-enabled `.venv-gpu` environment.
- Threshold selected via FAR/FRR balancing; adjust to match business tolerances.

---

## Highlights

- **Signer-Aware Pairing**: Clean signer ID extraction ensures robust positive pairs and hard negatives without cross-writer leakage.
- **GPU-Accelerated Training**: Mixed precision (AMP), cosine learning-rate schedule, gradient clipping, and pin-memory DataLoaders.
- **ResNet18 Backbone**: ImageNet-pretrained weights adapted to single-channel signatures with an L2-normalised projection head.
- **Dynamic Threshold Search**: Evaluation sweeps thresholds to minimise |FAR − FRR| and reports the full confusion matrix.
- **Reproducibility Built-In**: Deterministic seeds across Python, NumPy, and PyTorch plus CLI overrides for rapid experiments.

---

## Repository Layout

| Path | Description |
|------|-------------|
| `config.py` | Central hyperparameters, device selection, and path configuration. |
| `preprocess.py` | Converts `raw/full_org` and `raw/full_forg` assets into standardised grayscale crops in `data/processed/`. |
| `dataset.py` | Pairwise dataset with signer-aware sampling, augmentation, and deterministic evaluation transforms. |
| `model.py` | Siamese network wrapping ResNet18 with a projection head and L2 normalisation. |
| `loss.py` | Contrastive loss parameterised by margin. |
| `train.py` | End-to-end training loop (AMP, scheduler, early stopping, metrics, checkpointing). |
| `evaluate.py` | Loads `checkpoints/best_model.pth`, scores pairs, and outputs balanced-threshold metrics. |
| `utils.py` | Helper utilities for FAR/FRR, accuracy, ROC AUC, and confusion matrices. |

---

## Data Workflow

1. **Raw Inputs**: Store genuine signatures under `raw/full_org/` and forged signatures under `raw/full_forg/`. Filenames must contain a numeric signer identifier (e.g., `original_16_21.png`, `forgeries_10_8.png`).
2. **Preprocessing**: Run `python preprocess.py` to create resized grayscale copies under `data/processed/genuine/` and `data/processed/forged/`.
3. **Signer Splits**: `dataset.py` extracts numeric signer IDs, ensuring train/validation partitions remain writer-disjoint.

---

## Environment Setup (PowerShell)

```powershell
# Clone the repository
git clone https://github.com/Snikitha-V/Authentica.git
cd Authentica

# Create and activate the CUDA-enabled virtual environment (Python 3.12 suggested)
python -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The project has been validated with PyTorch 2.6.0 + CUDA 12.4 wheels on an NVIDIA GeForce MX550. Adjust package versions if deploying to different hardware.

---

## Preprocessing

```powershell
.\.venv-gpu\Scripts\Activate.ps1
python preprocess.py
```

- Non-image files are skipped safely.
- Outputs overwrite previous runs to guarantee a consistent training dataset.

---

## Training

```powershell
# Default configuration
python train.py

# Override common hyperparameters
python train.py --epochs 40 --batch-size 16 --lr 5e-4
```

- **AMP**: Controlled by `USE_AMP` in `config.py`; enabled automatically when CUDA is available.
- **Scheduler**: `CosineAnnealingLR` spans the requested epoch budget.
- **Early Stopping**: Monitors validation AUC with `PATIENCE = 10` (configurable).
- **Checkpointing**: Highest-validation-AUC weights saved to `checkpoints/best_model.pth`.

---

## Evaluation

```powershell
python evaluate.py
```

- Loads the best checkpoint, pushes inference to the configured device, and evaluates all pairs using deterministic transforms.
- Sweeps 100 thresholds between zero and the maximum distance, picks the one that balances FAR and FRR, and prints the resulting metrics and confusion matrix.

### Metric Primer

- **Accuracy**: Overall correctness at the selected threshold.
- **AUC**: Threshold-independent separation between genuine and forged embeddings (1.0 is ideal).
- **FAR**: Fraction of forged signatures incorrectly accepted as genuine.
- **FRR**: Fraction of genuine signatures incorrectly rejected as forged.

---

## Configuration Cheatsheet (`config.py`)

| Key | Default | Description |
|-----|---------|-------------|
| `DATA_DIR_GENUINE`, `DATA_DIR_FORGED` | `data/processed/...` | Locations of preprocessed images. |
| `IMG_SIZE` | `(155, 220)` | Height × width for resizing. |
| `BATCH_SIZE` | `32` | Default batch size (override via CLI). |
| `LEARNING_RATE` | `5e-4` | Adam learning rate. |
| `NUM_EPOCHS` | `50` | Max epochs before early stopping. |
| `EMBEDDING_DIM` | `256` | Output feature size of the Siamese branches. |
| `CONTRASTIVE_MARGIN` | `2.0` | Margin for the contrastive loss. |
| `USE_AMP` | `True` | Enables mixed precision on CUDA. |
| `CLIP_GRAD_NORM` | `1.0` | Gradient clipping threshold. |
| `SEED` | `42` | Global reproducibility seed. |

---

## Checkpoints & Deployment

1. Best weights are written to `checkpoints/best_model.pth` after each epoch that improves validation AUC.
2. Record the chosen inference threshold (e.g., `0.021`) alongside the checkpoint for deployment consistency.
3. Export to TorchScript or ONNX as needed (utility script forthcoming).

---

## Troubleshooting

- **Missing Samples**: Ensure preprocessing ran successfully and that filenames contain numeric signer IDs; otherwise, samples are filtered out.
- **CUDA Not Detected**: Confirm the `.venv-gpu` environment is active and run `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`.
- **Metrics Plateau**: Try different seeds, adjust `--batch-size`, or fine-tune the learning rate for your hardware.
- **Logging Needs**: Extend `train.py` with TensorBoard/CSV logging to monitor long training sessions.

---

## Roadmap

1. Automated ONNX export and inference harness.
2. CI pipeline for preprocessing, forward-pass smoke tests, and evaluation checks.
3. Advanced augmentation research (pressure simulation, elastic warping).

---

## License & Credits

- Powered by PyTorch, Torchvision, and scikit-learn.
- Signature dataset alignment inspired by the CEDAR signature repository—ensure compliance with the dataset’s license when distributing.

