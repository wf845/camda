# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CAMDA (Contrastive Attention-based Multi-view Deep Association) is a PyTorch-based machine learning research project for disease-miRNA association prediction using Graph Convolutional Networks (GCNs) and mutual information maximization.

## Common Development Commands

### Training and Evaluation
```bash
# Run full 5-fold cross-validation training
python code/evaluate.py

# Custom training with specific parameters
python code/evaluate.py --epochs 400 --hid_feats 256 --lr 0.001 --lambda_mi 0.5

# Different MLP predictor configurations
python code/evaluate.py --mlp_preset light     # Lightweight: [128, 1]
python code/evaluate.py --mlp_preset standard  # Standard: [256, 128, 1]  
python code/evaluate.py --mlp_preset deep      # Deep: [512, 256, 128, 64, 1]
python code/evaluate.py --mlp_preset linear    # Linear: [1]

# Custom MLP layers
python code/evaluate.py --mlp_layers 512 256 128 1 --mlp_dropout 0.4

# Disable mutual information loss (pure link prediction)
python code/evaluate.py --disable_mi_loss

# Enable cosine alignment between summary vectors
python code/evaluate.py --use_cosine_alignment true
```

### Dependencies
The project requires:
- PyTorch with CUDA support (GPU required)
- sklearn, matplotlib, pandas, numpy
- No package.json or requirements.txt found - dependencies must be installed manually

## High-Level Architecture

### Core Components

1. **CAMDA Model** (`code/camda.py`): Main neural network architecture
   - **GCNLayer/GCN**: Graph Convolutional Network layers with normalization and residual connections
   - **Attention**: Metapath-specific embedding fusion mechanism  
   - **Discriminator**: Contrastive learning component using bilinear discriminator
   - **ConfigurableMLP**: Multi-layer perceptron for link prediction with configurable architecture
   - **TopologyView**: Third-view encoder using personalized PageRank-based topology matrices
   - **CAMDA**: Main model combining 3 views (personalized, shared, topology) with attention fusion

2. **Training Pipeline** (`code/train.py`): End-to-end joint training system
   - Joint loss combining mutual information maximization and link prediction
   - 3-view MI maximization: 6 losses = 3 views × (Nodal + Global)  
   - Cross-validation with balanced positive/negative sampling
   - Dynamic threshold optimization for F1-score
   - Visualization generation (ROC/PR curves, cross-validation plots)

3. **Data Processing** (`code/utils.py`): Dataset loading and graph construction
   - Loads HMDD v2.0/v3.2 disease-miRNA association datasets
   - Builds metapath adjacency matrices: MDM, MMM, DMD, DDD, MDDM, DMMD
   - KNN-based sparsification of similarity matrices
   - GIP kernel computation for missing similarities
   - Fold-specific graph construction to prevent data leakage

4. **Main Entry Point** (`code/evaluate.py`): Command-line interface and configuration

### Data Flow

1. **Data Loading**: Load similarity matrices and known associations from `data/HMDD v*/`
2. **Graph Construction**: Build metapath adjacency matrices with KNN sparsification  
3. **Feature Integration**: Combine functional/semantic similarities with GIP kernel
4. **Model Training**: Joint optimization of 3-view encoders + discriminators + predictor
5. **Evaluation**: Cross-validation with ROC/PR analysis and visualization

### Key Architecture Patterns

- **Three-View Learning**: Personalized (Z), Shared (H), and Topology (S) representations
- **Metapath-Based**: Uses 6 metapaths (MDM, MMM, DMD, DDD, MDDM, DMMD) for different relationship patterns
- **Mutual Information Maximization**: 6 MI losses ensuring view consistency at nodal and global levels
- **End-to-End Training**: Joint optimization prevents representation collapse
- **Attention Fusion**: Learnable attention weights for metapath combination
- **Configurable Predictor**: MLP with presets (light/standard/deep/linear) or custom architecture

### Model Components Integration

- Each view (personalized/shared/topology) has separate GCN encoders for each metapath
- Attention mechanisms fuse metapath-specific representations within each view
- Discriminators enforce mutual information constraints between representations and summaries
- Final prediction uses concatenated representations from all three views
- Cross-validation ensures robust evaluation with fold-specific graph construction

## Dataset Structure

```
data/HMDD v2.0/ or data/HMDD v3.2/
├── disease number.txt
├── disease semantic similarity matrix 1.txt  
├── disease semantic similarity matrix 2.txt
├── disease semantic similarity weight matrix.txt
├── known disease-miRNA association number.txt
├── miRNA functional similarity matrix.txt
├── miRNA functional similarity weight matrix.txt
└── miRNA number.txt
```

## Important Implementation Details

- **GPU Required**: Model enforces CUDA availability, no CPU fallback
- **Fold-Specific Graphs**: Metapath matrices rebuilt per fold using only training associations
- **KNN Sparsification**: Default k=10 for miRNA/disease similarity matrices
- **Balanced Sampling**: 1:1 positive/negative ratio for link prediction
- **Dynamic Thresholds**: F1-optimal threshold found via grid search
- **Loss Balance**: λ parameter controls nodal vs global MI loss weighting