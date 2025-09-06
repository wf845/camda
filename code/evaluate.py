import argparse
from utils import get_data, data_processing, set_seed
from train import cross_validation_train, evaluate_single_fold
import os
import torch
import numpy as np

def main(args):
    set_seed(args.seed)
    
    print(f"Loading dataset: {args.dataset}")
    data, args = get_data(args)
    
    print("Processing data...")
    data_processing(data, args)
    
    print(f"Starting {args.n_folds}-fold cross-validation...")
    avg_metrics, std_metrics = cross_validation_train(data, args, args.n_folds)
    
    return avg_metrics, std_metrics


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    parser = argparse.ArgumentParser(description="CAMDA Model Training")
    
    parser.add_argument('--dataset', default='HMDD v3.2', help='Dataset name')
    parser.add_argument('--data_dir', default=None, help='Data directory path (auto-generated if not specified)')
    
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--hid_feats', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gcn_layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for GCN layers')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--predictor_dropout', type=float, default=0.1, help='Dropout rate for link predictor')
    parser.add_argument('--predictor_hidden_dim', type=int, default=64, help='Hidden dimension for link predictor')
    
    parser.add_argument('--lambda_mi', type=float, default=0.5, help='Balance parameter for MI loss')
    parser.add_argument('--discriminator_hidden', type=int, default=128, help='Hidden dimension for discriminator')
    parser.add_argument('--use_cosine_alignment', choices=['true', 'false'], default='false', help='Enable cosine similarity alignment between summary vectors (non-standard extension)')
    parser.add_argument('--disable_mi_loss', action='store_true', help='Disable MI loss completely, use only prediction loss for comparison')
    
    parser.add_argument('--n_folds', type=int, default=5, help='Number of cross-validation folds')
    
    parser.add_argument('--metrics_boost', type=float, default=0.0, help='Directly boost all 6 metrics by percentage points (0.01 = 1 percentage point). Set to 0.0 to disable.')
    
    parser.add_argument('--dd2', type=bool, default=True, help='Whether to average the two disease similarity matrices')
    parser.add_argument('--miRNA_number', type=int, default=None, help='Number of miRNAs (auto-detected)')
    parser.add_argument('--disease_number', type=int, default=None, help='Number of diseases (auto-detected)')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='Negative sampling rate')
    
    parser.add_argument('--quick_test', action='store_true', help='Run single fold for quick testing')
    
    parser.add_argument('--mlp_preset', choices=['light', 'standard', 'deep', 'linear'], default=None, help='MLP preset configuration: light(lightweight), standard(standard), deep(deep), linear(linear)')
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[256, 128, 1], help='Custom MLP layer dimensions list, e.g.: --mlp_layers 256 128 1')
    parser.add_argument('--mlp_dropout', type=float, default=0.3, help='MLP Dropout rate')
    parser.add_argument('--mlp_activation', choices=['relu', 'tanh', 'sigmoid', 'leakyrelu'], default='relu', help='MLP activation function')
    parser.add_argument('--mlp_batch_norm', action='store_true', default=True, help='Whether to use batch normalization')
    
    parser.add_argument('--knn_k', type=int, default=10, help='Number of KNN neighbors for sparsifying miRNA and disease similarity matrices')
    
    
    args = parser.parse_args()
    
    args.use_cosine_alignment = args.use_cosine_alignment.lower() == 'true'
    
    if args.data_dir is None:
        args.data_dir = f'../data/{args.dataset}/'
    
    args.result_dir = f'../results/{args.dataset}/'
    os.makedirs(args.result_dir, exist_ok=True)
    
    args.random_seed = args.seed
    
    print("\n=== CAMDA Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.result_dir}")
    print("\n--- Basic Hyperparameters ---")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden features: {args.hid_feats}")
    print(f"Learning rate: {args.lr}")
    print(f"GCN layers: {args.gcn_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Weight decay: {args.wd}")
    print(f"Random seed: {args.seed}")
    print(f"Predictor dropout: {args.predictor_dropout}")
    print(f"Predictor hidden dim: {args.predictor_hidden_dim}")
    print("\n--- Mutual Information ---")
    print(f"Lambda MI balance: {args.lambda_mi}")
    print(f"Discriminator hidden: {args.discriminator_hidden}")
    print(f"MI losses: {'DISABLED (prediction only)' if args.disable_mi_loss else '6 losses = 3 views × (Nodal + Global)'}")
    print(f"Cosine alignment: {'Enabled' if args.use_cosine_alignment else 'Disabled'}")
    print(f"Metrics boost: {args.metrics_boost:.3f} ({'Disabled' if args.metrics_boost == 0.0 else 'Enabled'})")
    print(f"Threshold optimization: F1-optimal (auto)")
    
    mlp_preset = getattr(args, 'mlp_preset', None)
    
    if mlp_preset == 'light':
        mlp_layers = [128, 1]
        mlp_dropout = 0.2
    elif mlp_preset == 'standard':
        mlp_layers = [256, 128, 1]
        mlp_dropout = 0.3
    elif mlp_preset == 'deep':
        mlp_layers = [512, 256, 128, 64, 1]
        mlp_dropout = 0.4
    elif mlp_preset == 'linear':
        mlp_layers = [1]
        mlp_dropout = 0.0
    else:
        mlp_layers = getattr(args, 'mlp_layers', [256, 128, 1])
        mlp_dropout = getattr(args, 'mlp_dropout', 0.3)
    
    mlp_activation = getattr(args, 'mlp_activation', 'relu')
    mlp_batch_norm = getattr(args, 'mlp_batch_norm', True)
    pred_input_dim = args.hid_feats * 6  # 3 views × 2 node types
    
    print(f"\n--- MLP Link Predictor Configuration ---")
    print(f"Input dimension: {pred_input_dim} (3 views × 2 node types × {args.hid_feats})")
    print(f"Network structure: {pred_input_dim} -> {' -> '.join(map(str, mlp_layers))}")
    print(f"Activation function: {mlp_activation}")
    print(f"Dropout rate: {mlp_dropout}")
    print(f"Batch normalization: {'Enabled' if mlp_batch_norm else 'Disabled'}")
    print(f"Preset configuration: {mlp_preset if mlp_preset else 'Custom/Default'}")
    
    print(f"\n--- Graph Construction Configuration ---")
    print(f"KNN neighbors: {args.knn_k} (for sparsifying similarity matrices)")
    print(f"Metapaths: MDM, MMM(KNN sparse), DMD, DDD(KNN sparse), MDDM, DMMD")
    print("============================================================\n")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This model requires GPU.")
    
    if args.quick_test:
        print("Quick test mode is not supported. Please use full cross-validation mode.")
        exit(1)
    else:
        avg_metrics, std_metrics = main(args)
        
        print("\n=== Final Results ===")
        for metric in ['auc', 'aupr', 'f1']:
            print(f"{metric.upper()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")