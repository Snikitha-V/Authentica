# Authentica

A Siamese signature verification system using PyTorch. Includes:

- GPU-enabled training with mixed precision
- ResNet18 backbone for 1-channel signatures
- Contrastive loss with configurable margin
- Strong data augmentation and hard negative sampling
- Optimal threshold search (FAR ≈ FRR) at evaluation

## Quickstart (PowerShell)

```powershell
.\.venv-gpu\Scripts\Activate.ps1
python preprocess.py
python train.py
python evaluate.py
```

## Notes
- Trained checkpoints are saved to `checkpoints/best_model.pth`.
- Raw data expected under `raw/full_org` and `raw/full_forg`; processed images under `data/processed/...`.
- See `config.py` for hyperparameters.
