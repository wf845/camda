# CAMDA: Contrastive Attention-based Multi-view Deep Association

## Description

CAMDA is a PyTorch framework for disease–miRNA association prediction. It learns three complementary views (personalized, shared, topology) from heterogeneous similarity graphs via GCNs and attention, and uses contrastive objectives plus an MLP predictor to score disease–miRNA pairs.

## Dataset Information

- Supported: HMDD v2.0, HMDD v3.2 (default).
- Expected layout:
```
data/
└── HMDD v3.2/  (or HMDD v2.0/)
    ├── disease number.txt
    ├── disease semantic similarity matrix 1.txt
    ├── disease semantic similarity matrix 2.txt
    ├── disease semantic similarity weight matrix.txt
    ├── known disease-miRNA association number.txt
    ├── miRNA functional similarity matrix.txt
    ├── miRNA functional similarity weight matrix.txt
    └── miRNA number.txt
```

## Code Information

- `code/evaluate.py`: CLI entry; loads data, runs k-fold cross-validation.
- `code/train.py`: Joint training loop, metrics, and plotting.
- `code/camda.py`: Model components (GCN, attention, discriminator, MLP, CAMDA).
- `code/utils.py`: Data loading, preprocessing, metapath construction, helpers.

## Usage Instructions

1) Install dependencies:
```bash
pip install -r requirements.txt
```
2) Place dataset under `data/HMDD v3.2/` (or `data/HMDD v2.0/`).
3) Run cross-validation (from the `code/` directory):
```bash
cd code
python evaluate.py --dataset "HMDD v3.2"
```
4) Optional flags (examples):
```bash
python evaluate.py --epochs 400 --hid_feats 256 --lr 0.001 --n_folds 5
python evaluate.py --mlp_preset standard   # or: light | deep | linear
python evaluate.py --disable_mi_loss       # prediction loss only
python evaluate.py --use_cosine_alignment true
python evaluate.py --knn_k 15              # KNN sparsification
```
Notes:
- By default, `evaluate.py` reads data from `../data/<dataset>/` relative to `code/`. Use `--data_dir` to override.
- CUDA-capable GPU is required; CPU execution is not supported in this setup.

## Requirements

- Python 3.9+
- CUDA-capable GPU with compatible PyTorch build
- Python libraries (from `requirements.txt`):
  - torch==2.4.0
  - dgl==2.4.0
  - numpy==1.26.4
  - pandas==2.2.2
  - scikit-learn==1.7.0
  - scipy==1.13.1
  - matplotlib==3.8.4

## Methodology

- Data processing: load HMDD similarities and known associations; integrate functional/semantic similarity with GIP kernels; sparsify graphs via KNN; build metapath adjacencies (MDM, MMM, DMD, DDD, MDDM, DMMD).
- Model: per-view GCN encoders (personalized/shared/topology) with attention fusion to obtain Z, H, S; optional cosine alignment between summaries.
- Training: joint optimization of contrastive MI losses and binary link prediction via an MLP on concatenated embeddings; k-fold cross-validation; metrics include AUC and AUPR.

## Citation

If you use this code or dataset, please cite:
```bibtex
@article{camda2024,
  title={CAMDA: Contrastive Attention-based Multi-view Deep Association for Disease-miRNA Association Prediction},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024},
  note={Available at: https://github.com/[username]/camda}
}
```

## License & Contribution Guidelines

- License: please include your chosen license in a `LICENSE` file.
- Contributions: open issues or pull requests with focused changes and clear descriptions.
# camda
