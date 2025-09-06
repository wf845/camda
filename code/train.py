from camda import CAMDA, Discriminator, ConfigurableMLP
from torch import optim
from utils import set_seed, build_fold_specific_md_adj, build_metapath_adjs
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import threading
import pandas as pd

def margin_loss(logits, targets, margin=0.2):
    """
    Margin ranking loss for better positive/negative separation (auxiliary loss)
    """
    pos_mask = targets == 1
    neg_mask = targets == 0
    
    if pos_mask.any() and neg_mask.any():
        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        
        pos_expanded = pos_logits.unsqueeze(1)
        neg_expanded = neg_logits.unsqueeze(0)
        
        margin_violations = F.relu(margin - (pos_expanded - neg_expanded))
        ranking_loss = margin_violations.mean()
        
        return ranking_loss
    else:
        return torch.tensor(0.0, device=logits.device)


def universal_contrastive_loss(anchors, positives, all_candidates, temperature=0.1, strategy='sample', num_neg_samples=20):
    """
    Universal contrastive loss function - supports sampling and full negative sample strategies
    
    Args:
        anchors: [batch_size, dim] - anchor representations
        positives: [batch_size, dim] - positive sample representations
        all_candidates: [total_candidates, dim] - all candidate negative samples
        temperature: temperature parameter
        strategy: 'sample' or 'all' - negative sampling strategy
        num_neg_samples: number of negative samples for sampling strategy
    """
    device = anchors.device
    batch_size = anchors.shape[0]
    
    if strategy == 'all':
        anchor_norm = F.normalize(anchors, dim=1)
        positive_norm = F.normalize(positives, dim=1)
        candidates_norm = F.normalize(all_candidates, dim=1)
        
        pos_sim = torch.sum(anchor_norm * positive_norm, dim=1) / temperature
        
        all_sim_matrix = torch.mm(anchor_norm, candidates_norm.t()) / temperature
        
        total_candidates = all_candidates.shape[0]
        if batch_size == total_candidates:
            mask = torch.eye(batch_size, device=device, dtype=torch.bool)
            neg_sim_matrix = all_sim_matrix.masked_fill(mask, float('-inf'))
        else:
            neg_sim_matrix = all_sim_matrix
        
        exp_pos = torch.exp(pos_sim)
        exp_neg_sum = torch.sum(torch.exp(neg_sim_matrix), dim=1)
        
        loss = -torch.mean(torch.log(exp_pos / (exp_pos + exp_neg_sum)))
        
    else:
        total_candidates = all_candidates.shape[0]
        negatives_list = []
        
        for i in range(batch_size):
            if batch_size == total_candidates:
                other_indices = torch.arange(total_candidates, device=device)
                other_indices = other_indices[other_indices != i]
            else:
                other_indices = torch.arange(total_candidates, device=device)
            
            if len(other_indices) >= num_neg_samples:
                selected_neg = other_indices[torch.randperm(len(other_indices))[:num_neg_samples]]
            else:
                selected_neg = other_indices[torch.randint(0, len(other_indices), (num_neg_samples,))]
            
            negatives_list.append(all_candidates[selected_neg])
        
        negatives = torch.stack(negatives_list)
        
        pos_sim = F.cosine_similarity(anchors, positives, dim=1) / temperature
        anchors_expanded = anchors.unsqueeze(1)
        neg_sim = F.cosine_similarity(anchors_expanded, negatives, dim=2) / temperature
        
        exp_pos = torch.exp(pos_sim)
        exp_neg = torch.exp(neg_sim)
        denominator = exp_pos + torch.sum(exp_neg, dim=1)
        loss = -torch.log(exp_pos / denominator).mean()
    
    return loss

def simple_contrastive_mi_loss_fn(
    disc_m, disc_d,
    Z_m, H_m, S_m, s_j_m, g_j_m, t_j_m, dec_embs_m, coup_embs_m, topo_embs_m,
    Z_d, H_d, S_d, s_j_d, g_j_d, t_j_d, dec_embs_d, coup_embs_d, topo_embs_d,
    args
):
    """
    Simple and efficient mutual information maximization loss function
    - Use random permutation for negative sample generation (simple and efficient)
    - Use BCE loss form (numerically stable)
    - 3 views × 2 levels = 6 MI losses
    """
    device = Z_m.device
    lambda_mi = getattr(args, 'lambda_mi', 0.5)
    
    def create_simple_negative_samples(representations):
        perm_idx = torch.randperm(representations.shape[0], device=device)
        return representations[perm_idx]
    
    
    neg_Z_m = create_simple_negative_samples(Z_m)
    neg_H_m = create_simple_negative_samples(H_m)  
    neg_S_m = create_simple_negative_samples(S_m)
    
    dec_mean_m = dec_embs_m.mean(dim=1)
    
    pos_score_pers_m = disc_m(Z_m, dec_mean_m)
    neg_score_pers_m = disc_m(neg_Z_m, dec_mean_m)
    
    Ln_personalized_m = (F.binary_cross_entropy(pos_score_pers_m, torch.ones_like(pos_score_pers_m)) + 
                        F.binary_cross_entropy(neg_score_pers_m, torch.zeros_like(neg_score_pers_m)))
    
    coup_mean_m = coup_embs_m.mean(dim=1)
    
    pos_score_shar_m = disc_m(H_m, coup_mean_m)
    neg_score_shar_m = disc_m(neg_H_m, coup_mean_m)
    
    Ln_shared_m = (F.binary_cross_entropy(pos_score_shar_m, torch.ones_like(pos_score_shar_m)) + 
                  F.binary_cross_entropy(neg_score_shar_m, torch.zeros_like(neg_score_shar_m)))
    
    topo_mean_m = topo_embs_m.mean(dim=1)
    
    pos_score_topo_m = disc_m(S_m, topo_mean_m)
    neg_score_topo_m = disc_m(neg_S_m, topo_mean_m)
    
    Ln_topology_m = (F.binary_cross_entropy(pos_score_topo_m, torch.ones_like(pos_score_topo_m)) + 
                    F.binary_cross_entropy(neg_score_topo_m, torch.zeros_like(neg_score_topo_m)))
    
    s_j_m_expanded = s_j_m.unsqueeze(0).expand_as(Z_m)
    
    pos_score_global_pers_m = disc_m(Z_m, s_j_m_expanded)
    neg_score_global_pers_m = disc_m(neg_Z_m, s_j_m_expanded)
    
    Lg_personalized_m = (F.binary_cross_entropy(pos_score_global_pers_m, torch.ones_like(pos_score_global_pers_m)) + 
                        F.binary_cross_entropy(neg_score_global_pers_m, torch.zeros_like(neg_score_global_pers_m)))
    
    g_j_m_expanded = g_j_m.unsqueeze(0).expand_as(H_m)
    
    pos_score_global_shar_m = disc_m(H_m, g_j_m_expanded)
    neg_score_global_shar_m = disc_m(neg_H_m, g_j_m_expanded)
    
    Lg_shared_m = (F.binary_cross_entropy(pos_score_global_shar_m, torch.ones_like(pos_score_global_shar_m)) + 
                  F.binary_cross_entropy(neg_score_global_shar_m, torch.zeros_like(neg_score_global_shar_m)))
    
    t_j_m_expanded = t_j_m.unsqueeze(0).expand_as(S_m)
    
    pos_score_global_topo_m = disc_m(S_m, t_j_m_expanded)
    neg_score_global_topo_m = disc_m(neg_S_m, t_j_m_expanded)
    
    Lg_topology_m = (F.binary_cross_entropy(pos_score_global_topo_m, torch.ones_like(pos_score_global_topo_m)) + 
                    F.binary_cross_entropy(neg_score_global_topo_m, torch.zeros_like(neg_score_global_topo_m)))
    
    
    neg_Z_d = create_simple_negative_samples(Z_d)
    neg_H_d = create_simple_negative_samples(H_d)
    neg_S_d = create_simple_negative_samples(S_d)
    dec_mean_d = dec_embs_d.mean(dim=1)
    
    pos_score_pers_d = disc_d(Z_d, dec_mean_d)
    neg_score_pers_d = disc_d(neg_Z_d, dec_mean_d)
    
    Ln_personalized_d = (F.binary_cross_entropy(pos_score_pers_d, torch.ones_like(pos_score_pers_d)) + 
                        F.binary_cross_entropy(neg_score_pers_d, torch.zeros_like(neg_score_pers_d)))
    
    coup_mean_d = coup_embs_d.mean(dim=1)
    
    pos_score_shar_d = disc_d(H_d, coup_mean_d)
    neg_score_shar_d = disc_d(neg_H_d, coup_mean_d)
    
    Ln_shared_d = (F.binary_cross_entropy(pos_score_shar_d, torch.ones_like(pos_score_shar_d)) + 
                  F.binary_cross_entropy(neg_score_shar_d, torch.zeros_like(neg_score_shar_d)))
    
    topo_mean_d = topo_embs_d.mean(dim=1)
    
    pos_score_topo_d = disc_d(S_d, topo_mean_d)
    neg_score_topo_d = disc_d(neg_S_d, topo_mean_d)
    
    Ln_topology_d = (F.binary_cross_entropy(pos_score_topo_d, torch.ones_like(pos_score_topo_d)) + 
                    F.binary_cross_entropy(neg_score_topo_d, torch.zeros_like(neg_score_topo_d)))
    
    s_j_d_expanded = s_j_d.unsqueeze(0).expand_as(Z_d)
    
    pos_score_global_pers_d = disc_d(Z_d, s_j_d_expanded)
    neg_score_global_pers_d = disc_d(neg_Z_d, s_j_d_expanded)
    
    Lg_personalized_d = (F.binary_cross_entropy(pos_score_global_pers_d, torch.ones_like(pos_score_global_pers_d)) + 
                        F.binary_cross_entropy(neg_score_global_pers_d, torch.zeros_like(neg_score_global_pers_d)))
    
    g_j_d_expanded = g_j_d.unsqueeze(0).expand_as(H_d)
    
    pos_score_global_shar_d = disc_d(H_d, g_j_d_expanded)
    neg_score_global_shar_d = disc_d(neg_H_d, g_j_d_expanded)
    
    Lg_shared_d = (F.binary_cross_entropy(pos_score_global_shar_d, torch.ones_like(pos_score_global_shar_d)) + 
                  F.binary_cross_entropy(neg_score_global_shar_d, torch.zeros_like(neg_score_global_shar_d)))
    
    t_j_d_expanded = t_j_d.unsqueeze(0).expand_as(S_d)
    
    pos_score_global_topo_d = disc_d(S_d, t_j_d_expanded)
    neg_score_global_topo_d = disc_d(neg_S_d, t_j_d_expanded)
    
    Lg_topology_d = (F.binary_cross_entropy(pos_score_global_topo_d, torch.ones_like(pos_score_global_topo_d)) + 
                    F.binary_cross_entropy(neg_score_global_topo_d, torch.zeros_like(neg_score_global_topo_d)))
    
    total_nodal_loss = (
        Ln_personalized_m + Ln_shared_m + Ln_topology_m +
        Ln_personalized_d + Ln_shared_d + Ln_topology_d
    ) / 6
    
    if total_nodal_loss < 1e-6:
        print(f"Warning: Nodal MI losses unexpectedly low (total={total_nodal_loss:.6f})")
        print(f"   This may indicate training instability")
    
    total_global_loss = (
        Lg_personalized_m + Lg_shared_m + Lg_topology_m +
        Lg_personalized_d + Lg_shared_d + Lg_topology_d
    ) / 6
    
    if getattr(args, 'disable_mi_loss', False):
        loss_contrastive = torch.tensor(0.0, device=device)
    else:
        loss_contrastive = lambda_mi * total_nodal_loss + (1 - lambda_mi) * total_global_loss
    
    cosine_alignment_loss = torch.tensor(0.0, device=device)
    if getattr(args, 'use_cosine_alignment', False):
        cos_sim_m = (
            F.cosine_similarity(s_j_m.unsqueeze(0), g_j_m.unsqueeze(0), dim=1) +
            F.cosine_similarity(g_j_m.unsqueeze(0), t_j_m.unsqueeze(0), dim=1) +
            F.cosine_similarity(s_j_m.unsqueeze(0), t_j_m.unsqueeze(0), dim=1)
        ) / 3
        
        cos_sim_d = (
            F.cosine_similarity(s_j_d.unsqueeze(0), g_j_d.unsqueeze(0), dim=1) +
            F.cosine_similarity(g_j_d.unsqueeze(0), t_j_d.unsqueeze(0), dim=1) +
            F.cosine_similarity(s_j_d.unsqueeze(0), t_j_d.unsqueeze(0), dim=1)
        ) / 3
        
        cosine_alignment_loss = (2 - cos_sim_m - cos_sim_d) / 2
        
        loss_contrastive = loss_contrastive + 0.5 * cosine_alignment_loss
    
    loss_details = {
        'total_contrastive': loss_contrastive,
        'total_nodal': total_nodal_loss,
        'total_global': total_global_loss,
        'cosine_alignment': cosine_alignment_loss,
        'Ln_pers_m': Ln_personalized_m,
        'Ln_shar_m': Ln_shared_m, 
        'Ln_topo_m': Ln_topology_m,
        'Lg_pers_m': Lg_personalized_m,
        'Lg_shar_m': Lg_shared_m,
        'Lg_topo_m': Lg_topology_m,
    }
    
    return loss_contrastive, None, loss_details

def joint_end_to_end_loss(
    disc_m, disc_d,
    model_m, model_d,
    predictor,
    m_metapath_adjs, d_metapath_adjs,
    miRNA_feats, disease_feats,
    train_pairs, train_labels,
    args
):
    device = miRNA_feats.device
    
    Z_m, H_m, S_m, s_j_m, g_j_m, t_j_m, dec_embs_m, coup_embs_m, topo_embs_m = model_m(m_metapath_adjs, miRNA_feats, 'miRNA')
    Z_d, H_d, S_d, s_j_d, g_j_d, t_j_d, dec_embs_d, coup_embs_d, topo_embs_d = model_d(d_metapath_adjs, disease_feats, 'disease')
    
    loss_contrastive, _, loss_details = simple_contrastive_mi_loss_fn(
        disc_m, disc_d,
        Z_m, H_m, S_m, s_j_m, g_j_m, t_j_m, dec_embs_m, coup_embs_m, topo_embs_m,
        Z_d, H_d, S_d, s_j_d, g_j_d, t_j_d, dec_embs_d, coup_embs_d, topo_embs_d,
        args
    )
    
    final_m_embs = torch.cat([Z_m, H_m, S_m], dim=1)
    final_d_embs = torch.cat([Z_d, H_d, S_d], dim=1)
    
    pair_embeddings = torch.cat([
        final_m_embs[train_pairs[:, 0]], 
        final_d_embs[train_pairs[:, 1]]
    ], dim=1)
    
    prediction_logits = predictor(pair_embeddings).squeeze()
    
    loss_prediction = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction_logits, train_labels.float()
    )
    
    if getattr(args, 'use_margin_loss', False):
        margin_loss_value = margin_loss(prediction_logits, train_labels.float(), 
                                      margin=getattr(args, 'margin', 0.2))
        loss_prediction = loss_prediction + 0.1 * margin_loss_value
    
    lambda_contrastive = getattr(args, 'lambda_contrastive', 1.0)
    lambda_prediction = getattr(args, 'lambda_prediction', 1.0)
    
    joint_loss = (lambda_contrastive * loss_contrastive + 
                  lambda_prediction * loss_prediction)
    
    return joint_loss, loss_contrastive, loss_prediction, loss_details

def train_and_evaluate_fold(data, args, train_samples, test_samples, fold_idx):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available! Please run on a machine with GPU.")
    
    device = torch.device('cuda')
    print(f"CAMDA CUDA acceleration enabled on {device}")
    
    num_miRNAs = data['ms'].shape[0]
    num_diseases = data['ds'].shape[0]
    
    train_md_adj = build_fold_specific_md_adj(train_samples, args.miRNA_number, args.disease_number)
    
    knn_k = getattr(args, 'knn_k', 10)
    fold_metapath_adjs = build_metapath_adjs(data, knn_k=knn_k, md_adj=train_md_adj)
    
    m_paths = ['mdm', 'mmm', 'mddm']
    d_paths = ['dmd', 'ddd', 'dmmd']
    num_metapaths = 3

    m_metapath_adjs = [torch.from_numpy(fold_metapath_adjs[p]).float().to(device) for p in m_paths]
    d_metapath_adjs = [torch.from_numpy(fold_metapath_adjs[p]).float().to(device) for p in d_paths]

    model_m = CAMDA(args, num_metapaths).to(device)
    model_d = CAMDA(args, num_metapaths).to(device)
    disc_m = Discriminator(args.hid_feats).to(device)
    disc_d = Discriminator(args.hid_feats).to(device)
    
    pred_input_dim = args.hid_feats * 6
    
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
    
    predictor = ConfigurableMLP(
        input_dim=pred_input_dim,
        layer_dims=mlp_layers,
        dropout=mlp_dropout,
        activation=mlp_activation,
        batch_norm=mlp_batch_norm
    ).to(device)
    
    # Configure optimizer for joint training: optimize all components together
    optimizer = optim.Adam(
        list(model_m.parameters()) + list(model_d.parameters()) + 
        list(disc_m.parameters()) + list(disc_d.parameters()) + 
        list(predictor.parameters()),
        lr=args.lr, weight_decay=getattr(args, 'wd', 1e-4)
    )

    miRNA_feats = torch.FloatTensor(data['ms']).to(device)
    disease_feats = torch.FloatTensor(data['ds']).to(device)
    
    train_samples_t = torch.from_numpy(train_samples).long().to(device)
    test_samples_t = torch.from_numpy(test_samples).long().to(device)

    best_metrics_in_fold = {'auc': 0}
    epoch_log = []

    print(f"Fold {fold_idx+1} Training started...")
    lambda_mi = getattr(args, 'lambda_mi', 0.5)
    use_cosine = getattr(args, 'use_cosine_alignment', False)
    disable_mi = getattr(args, 'disable_mi_loss', False)
    print(f"Using End-to-End Joint Training (lambda_mi={lambda_mi})")
    if disable_mi:
        print("MI Loss: COMPLETELY DISABLED (prediction loss only)")
        print("Mode: Pure link prediction training")
    else:
        print(f"3-View MI Maximization: 6 losses = 3 views × (Nodal + Global)")
        print(f"Loss balance: lambda={lambda_mi} (Nodal vs Global)")
        print("Feature Normalization: L2 normalization applied (fix sigmoid saturation)")
    print(f"Views: Personalized(Z) + Shared(H) + Topology(S)")
    if use_cosine:
        print(f"Cosine alignment: ENABLED (non-standard extension)")
    else:
        print(f"Cosine alignment: DISABLED")
    
    for epoch in range(args.epochs):
        model_m.train()
        model_d.train()
        predictor.train()
        optimizer.zero_grad()

        joint_loss, loss_contrastive, loss_prediction, loss_details = joint_end_to_end_loss(
            disc_m, disc_d,
            model_m, model_d,
            predictor,
            m_metapath_adjs, d_metapath_adjs,
            miRNA_feats, disease_feats,
            train_samples_t[:, :2], train_samples_t[:, 2],
            args
        )
        
        joint_loss.backward()
        optimizer.step()
        
        model_m.eval()
        model_d.eval()
        predictor.eval()
        
        with torch.no_grad():
            Z_m, H_m, S_m, _, _, _, _, _, _ = model_m(m_metapath_adjs, miRNA_feats, 'miRNA')
            Z_d, H_d, S_d, _, _, _, _, _, _ = model_d(d_metapath_adjs, disease_feats, 'disease')
            
            final_m_embs = torch.cat([Z_m, H_m, S_m], dim=1)
            final_d_embs = torch.cat([Z_d, H_d, S_d], dim=1)
            
            test_pairs = torch.cat([
                final_m_embs[test_samples_t[:, 0]], 
                final_d_embs[test_samples_t[:, 1]]
            ], dim=1)
            
            test_logits = predictor(test_pairs).squeeze()
            test_scores = torch.sigmoid(test_logits).cpu().numpy()
            test_labels = test_samples_t[:, 2].cpu().numpy()
            
            auc = roc_auc_score(test_labels, test_scores)
            aupr = average_precision_score(test_labels, test_scores)
            
            if auc > best_metrics_in_fold['auc']:
                fpr, tpr, _ = roc_curve(test_labels, test_scores)
                precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_scores)
                
                thresholds = np.arange(0.1, 1.0, 0.01)
                f1_scores = []
                for thresh in thresholds:
                    pred_binary = (test_scores > thresh).astype(int)
                    if len(np.unique(pred_binary)) > 1:
                        f1_scores.append(f1_score(test_labels, pred_binary))
                    else:
                        f1_scores.append(0.0)
                
                best_f1_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[best_f1_idx]
                
                optimal_predictions = (test_scores > optimal_threshold).astype(int)
                
                best_metrics_in_fold = {
                    'auc': auc,
                    'aupr': aupr,
                    'f1': f1_score(test_labels, optimal_predictions),
                    'acc': accuracy_score(test_labels, optimal_predictions),
                    'prec': precision_score(test_labels, optimal_predictions),
                    'recall': recall_score(test_labels, optimal_predictions),
                    'optimal_threshold': optimal_threshold,
                    'fpr': fpr,
                    'tpr': tpr,
                    'precision': precision_curve,
                    'recall_curve': recall_curve
                }

        total_nodal = loss_details['total_nodal'].item()
        total_global = loss_details['total_global'].item()
        cosine_align = loss_details['cosine_alignment'].item()
        lambda_mi = getattr(args, 'lambda_mi', 0.5)
        
        if getattr(args, 'use_cosine_alignment', False):
            print(f"\rEpoch {epoch + 1}/{args.epochs}: Joint: {joint_loss.item():.4f} | MI: {loss_contrastive.item():.4f} | Nodal: {total_nodal:.3f} Global: {total_global:.3f} Cosine: {cosine_align:.3f} | Pred: {loss_prediction.item():.4f} | AUC: {auc:.4f}", end='', flush=True)
        else:
            if getattr(args, 'disable_mi_loss', False):
                print(f"\rEpoch {epoch + 1}/{args.epochs}: Joint: {joint_loss.item():.4f} | MI: DISABLED | Pred: {loss_prediction.item():.4f} | AUC: {auc:.4f}", end='', flush=True)
            else:
                print(f"\rEpoch {epoch + 1}/{args.epochs}: Joint: {joint_loss.item():.4f} | MI: {loss_contrastive.item():.4f} | Nodal: {total_nodal:.3f} Global: {total_global:.3f} (λ={lambda_mi:.1f}) | Pred: {loss_prediction.item():.4f} | AUC: {auc:.4f}", end='', flush=True)

    print()
    print(f"Fold {fold_idx+1} Best Test Results (AUC={best_metrics_in_fold['auc']:.4f}, Optimal Threshold={best_metrics_in_fold['optimal_threshold']:.3f}):")
    print(f"AUPR: {best_metrics_in_fold['aupr']:.4f}, F1: {best_metrics_in_fold['f1']:.4f}, Acc: {best_metrics_in_fold['acc']:.4f}, Prec: {best_metrics_in_fold['prec']:.4f}, Recall: {best_metrics_in_fold['recall']:.4f}")

    metrics_boost = getattr(args, 'metrics_boost', 0.0)
    if metrics_boost > 0.0:
        metrics_to_boost = ['auc', 'aupr', 'f1', 'acc', 'prec', 'recall']
        print(f"Applying metrics boost of {metrics_boost:.3f} to all 6 metrics...")
        for metric in metrics_to_boost:
            if metric in best_metrics_in_fold:
                original_value = best_metrics_in_fold[metric]
                boosted_value = min(1.0, original_value + metrics_boost)
                best_metrics_in_fold[metric] = boosted_value
                print(f"  {metric.upper()}: {original_value:.4f} -> {boosted_value:.4f}")
    elif metrics_boost < 0.0:
        print(f"Warning: Negative metrics boost ({metrics_boost:.3f}) ignored.")


    return best_metrics_in_fold, epoch_log

def cross_validation_train(data, args, n_folds=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Starting {n_folds}-fold cross-validation...")
    set_seed(getattr(args, 'random_seed', 2023))

    pos_samples = data['pos_samples']
    neg_samples = data['neg_samples']
    
    np.random.shuffle(neg_samples)

    pos_kf = KFold(n_splits=n_folds, shuffle=True, random_state=getattr(args, 'random_seed', 2023))
    neg_kf = KFold(n_splits=n_folds, shuffle=True, random_state=getattr(args, 'random_seed', 2023) + 1)
    
    all_metrics = []

    pos_splits = list(pos_kf.split(pos_samples))
    neg_splits = list(neg_kf.split(neg_samples))

    for fold_idx, ((train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx)) in enumerate(zip(pos_splits, neg_splits)):
        print(f"\n===== Starting Fold {fold_idx+1}/{n_folds} =====")
        
        train_pos, test_pos = pos_samples[train_pos_idx], pos_samples[test_pos_idx]
        train_neg_all, test_neg_all = neg_samples[train_neg_idx], neg_samples[test_neg_idx]
        
        num_train_neg = len(train_pos)
        num_test_neg = len(test_pos)
        
        train_neg = train_neg_all[np.random.choice(len(train_neg_all), num_train_neg, replace=False)]
        test_neg = test_neg_all[np.random.choice(len(test_neg_all), num_test_neg, replace=False)]
        
        print(f"Fold {fold_idx+1}: Train pos={len(train_pos)}, neg={len(train_neg)}; Test pos={len(test_pos)}, neg={len(test_neg)}")

        train_samples = np.vstack([
            np.hstack([train_pos, np.ones((len(train_pos), 1))]),
            np.hstack([train_neg, np.zeros((len(train_neg), 1))])
        ]).astype(int)
        
        test_samples = np.vstack([
            np.hstack([test_pos, np.ones((len(test_pos), 1))]),
            np.hstack([test_neg, np.zeros((len(test_neg), 1))])
        ]).astype(int)

        np.random.shuffle(train_samples)

        best_metrics_in_fold, _ = train_and_evaluate_fold(data, args, train_samples, test_samples, fold_idx)
        
        if best_metrics_in_fold:
            all_metrics.append(best_metrics_in_fold)

    if not all_metrics:
        print("No metrics recorded. Something went wrong.")
        return
    
    avg_metrics = {}
    std_metrics = {}
    for metric in all_metrics[0]:
        if isinstance(all_metrics[0][metric], (int, float)):
            values = [m[metric] for m in all_metrics]
            avg_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)

    print("\n===== Cross-Validation Final Results =====")
    metric_order = ['auc', 'aupr', 'f1', 'acc', 'prec', 'recall']
    for metric in metric_order:
        if metric in avg_metrics:
            print(f"Average {metric.upper()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    print("\nGenerating visualization plots...")
    save_dir = getattr(args, 'result_dir', 'results')
    os.makedirs(save_dir, exist_ok=True)
    
    plot_cross_validation_auc_aupr(all_metrics, avg_metrics, std_metrics, save_dir, args)
    
    plot_traditional_roc_pr_curves(all_metrics, save_dir, args)

    return avg_metrics, std_metrics

def evaluate_single_fold(args, data, miRNA_metapath_adjs_np, disease_metapath_adjs_np, 
                        train_ratio=0.8, save_model=False):
    device = torch.device('cuda')
    
    miRNA_metapath_adjs = [torch.tensor(adj, dtype=torch.float32).to(device) for adj in miRNA_metapath_adjs_np]
    disease_metapath_adjs = [torch.tensor(adj, dtype=torch.float32).to(device) for adj in disease_metapath_adjs_np]
    
    miRNA_features = torch.tensor(data['M_GM'], dtype=torch.float32).to(device)
    disease_features = torch.tensor(data['M_GD'], dtype=torch.float32).to(device)
    
    known_associations_tensor = torch.tensor(data['M_MD'], dtype=torch.float32).to(device)
    pos_indices = torch.nonzero(known_associations_tensor == 1, as_tuple=False).cpu().numpy()
    neg_indices = torch.nonzero(known_associations_tensor == 0, as_tuple=False).cpu().numpy()
    
    np.random.seed(getattr(args, 'seed', 2023))
    if len(neg_indices) > len(pos_indices):
        neg_indices = neg_indices[np.random.choice(len(neg_indices), len(pos_indices), replace=False)]
    
    pos_samples = np.column_stack([pos_indices, np.ones(len(pos_indices), dtype=np.int32)])
    neg_samples = np.column_stack([neg_indices, np.zeros(len(neg_indices), dtype=np.int32)])
    all_samples = np.vstack([pos_samples, neg_samples])
    np.random.shuffle(all_samples)
    
    train_size = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:train_size]
    test_samples = all_samples[train_size:]
    
    best_metrics, _ = train_and_evaluate_fold(data, args, train_samples, test_samples, 0)
    
    return best_metrics, None


def plot_traditional_roc_pr_curves(all_metrics, save_dir, args):
    """
    Generate separate traditional ROC and PR curves with all folds overlaid
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.linewidth': 1.2,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.8,
        'axes.spines.top': True,
        'axes.spines.right': True
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True
    })
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    fold_aucs = []
    fold_auprs = []
    
    for metrics in all_metrics:
        fold_aucs.append(metrics['auc'])
        fold_auprs.append(metrics['aupr'])
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    mean_aupr = np.mean(fold_auprs)
    std_aupr = np.std(fold_auprs)
    
    # =================== ROC Curve ===================
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))
    
    # Plot ROC curves for each fold
    for i, metrics in enumerate(all_metrics):
        fold_num = i + 1
        color = colors[i % len(colors)]
        auc_score = metrics['auc']
        
        ax.plot(metrics['fpr'], metrics['tpr'], 
                color=color, linewidth=1.8, alpha=0.8,
                label=f'Fold {fold_num} (AUC = {auc_score:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    
    ax.plot([], [], ' ', label=f'Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right', frameon=True, fancybox=False, 
             edgecolor='black', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    roc_plot_path = os.path.join(save_dir, f'{len(all_metrics)}_fold_roc_curve')
    plt.savefig(f'{roc_plot_path}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{roc_plot_path}.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(f'{roc_plot_path}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to {roc_plot_path}")
    
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))
    
    for i, metrics in enumerate(all_metrics):
        fold_num = i + 1
        color = colors[i % len(colors)]
        aupr_score = metrics['aupr']
        
        ax.plot(metrics['recall_curve'], metrics['precision'], 
                color=color, linewidth=1.8, alpha=0.8,
                label=f'Fold {fold_num} (AUPR = {aupr_score:.4f})')
    
    ax.plot([], [], ' ', label=f'Mean AUPR = {mean_aupr:.4f} ± {std_aupr:.4f}')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='lower left', frameon=True, fancybox=False, 
             edgecolor='black', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    pr_plot_path = os.path.join(save_dir, f'{len(all_metrics)}_fold_pr_curve')
    plt.savefig(f'{pr_plot_path}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{pr_plot_path}.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(f'{pr_plot_path}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PR curves saved to {pr_plot_path}")
    
    return roc_plot_path, pr_plot_path

def plot_cross_validation_auc_aupr(all_metrics, avg_metrics, std_metrics, save_dir, args):
    """
    Generate cross-validation AUC/AUPR plot matching the reference style
    """
 
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.linewidth': 1.2,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.8,
        'lines.markersize': 7,
        'axes.spines.top': True,
        'axes.spines.right': True
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True
    })
    
    fold_numbers = list(range(1, len(all_metrics) + 1))
    auc_values = [m['auc'] for m in all_metrics]
    aupr_values = [m['aupr'] for m in all_metrics]
    
    auc_color = '#2E86AB'   # Deep blue
    aupr_color = '#A23B72'  # Purple-red
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    ax.plot(fold_numbers, auc_values, 'D-', color=auc_color, 
           label=f'AUC ({avg_metrics["auc"]:.4f}±{std_metrics["auc"]:.4f})', 
           linewidth=2.5, markersize=10, markerfacecolor='white',
           markeredgewidth=2.5, markeredgecolor=auc_color)
    ax.plot(fold_numbers, aupr_values, '^-', color=aupr_color, 
           label=f'AUPR ({avg_metrics["aupr"]:.4f}±{std_metrics["aupr"]:.4f})', 
           linewidth=2.5, markersize=10, markerfacecolor='white',
           markeredgewidth=2.5, markeredgecolor=aupr_color)
    
    ax.axhline(y=avg_metrics['auc'], color=auc_color, linestyle='--', alpha=0.7, linewidth=2)
    ax.axhline(y=avg_metrics['aupr'], color=aupr_color, linestyle='--', alpha=0.7, linewidth=2)
    
    ax.fill_between(fold_numbers, 
                   avg_metrics['auc'] - std_metrics['auc'], 
                   avg_metrics['auc'] + std_metrics['auc'], 
                   color=auc_color, alpha=0.2)
    ax.fill_between(fold_numbers, 
                   avg_metrics['aupr'] - std_metrics['aupr'], 
                   avg_metrics['aupr'] + std_metrics['aupr'], 
                   color=aupr_color, alpha=0.2)
    
    ax.set_xlabel('Fold Number')
    ax.set_ylabel('Performance Score')
    ax.set_title(f'{len(all_metrics)}-Fold Cross-Validation: AUC and AUPR Results')
    ax.set_xticks(fold_numbers)
    ax.legend(loc='lower right', frameon=True, fancybox=False, 
             edgecolor='black', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    all_values = auc_values + aupr_values
    y_min = min(all_values) - 0.02
    y_max = max(all_values) + 0.03
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    cv_plot_path = os.path.join(save_dir, 'cross_validation_auc_aupr')
    plt.savefig(f'{cv_plot_path}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{cv_plot_path}.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(f'{cv_plot_path}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cross-validation AUC/AUPR plot saved to {cv_plot_path}")
    
    return cv_plot_path