import dgl
import math
import random
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def set_seed(seed):
    """
    Set random seed to ensure reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def compute_similarity(matrix):
    """
    Compute cosine similarity matrix of feature matrix
    """
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0)
    return similarity

def get_edge_index(matrix, device=None):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    if device is not None:
        return th.LongTensor(edge_index).to(device)
    return th.LongTensor(edge_index)


def build_fold_specific_md_adj(train_samples, miRNA_number, disease_number):
    """
    Build miRNA-disease association matrix based on training set samples
    
    Args:
        train_samples: training set samples, format [[miRNA_idx, disease_idx, label], ...]
        miRNA_number: total number of miRNAs
        disease_number: total number of diseases
    
    Returns:
        md_adj matrix
    """
    md_adj = np.zeros((miRNA_number, disease_number))
    
    for sample in train_samples:
        miRNA_idx, disease_idx, label = sample
        if label == 1:
            md_adj[miRNA_idx, disease_idx] = 1
    
    return md_adj

def build_metapath_adjs(data, binarize=True, knn_k=10, md_adj=None):
    """
    Build adjacency matrices according to defined metapaths.
    Metapaths:
    - M-D-M: miRNA -> Disease -> miRNA
    - M-M-M: miRNA -> miRNA -> miRNA (based on KNN sparsification)
    - D-M-D: Disease -> miRNA -> Disease
    - D-D-D: Disease -> Disease -> Disease (based on KNN sparsification)
    
    Args:
        data: data dictionary
        binarize: whether to binarize
        knn_k: KNN neighbor count, used for sparsifying similarity matrices, default 10
        md_adj: optional miRNA-disease association matrix, if provided use this matrix, otherwise use data['md_adj']
    """
    if md_adj is not None:
        A_md = md_adj
    else:
        A_md = data['md_adj'].numpy() if isinstance(data['md_adj'], th.Tensor) else data['md_adj']
    A_mm_original = data['mf']
    A_dd_original = data['dss']
    
    print(f"Using KNN(k={knn_k}) to sparsify similarity matrices...")
    A_mm = k_matrix(A_mm_original, k=knn_k)
    A_dd = k_matrix(A_dd_original, k=knn_k)
    
    mm_density = np.sum(A_mm > 0) / (A_mm.shape[0] * A_mm.shape[1])
    dd_density = np.sum(A_dd > 0) / (A_dd.shape[0] * A_dd.shape[1])
    print(f"   miRNA graph sparsity: {mm_density:.4f}")
    print(f"   Disease graph sparsity: {dd_density:.4f}")

    adj_mdm = A_md @ A_md.T
    adj_mmm = A_mm
    adj_dmd = A_md.T @ A_md
    adj_ddd = A_dd
    
    adj_mddm = A_md @ A_dd @ A_md.T
    
    adj_dmmd = A_md.T @ A_mm @ A_md

    metapath_adjs = {
        'mdm': adj_mdm,
        'mmm': adj_mmm,
        'dmd': adj_dmd,
        'ddd': adj_ddd,
        'mddm': adj_mddm,
        'dmmd': adj_dmmd
    }

    if binarize:
        for key, matrix in metapath_adjs.items():
            binarized_matrix = np.where(matrix > 0, 1, 0)
            np.fill_diagonal(binarized_matrix, 0)
            metapath_adjs[key] = binarized_matrix
            
    return metapath_adjs


def make_adj(edges, size, device=None):
    if device is not None:
        edges_tensor = th.LongTensor(edges).t().to(device)
        values = th.ones(len(edges)).to(device)
        adj = th.sparse_coo_tensor(edges_tensor, values, size, device=device).to_dense().long()
    else:
        edges_tensor = th.LongTensor(edges).t()
        values = th.ones(len(edges))
        adj = th.sparse_coo_tensor(edges_tensor, values, size).to_dense().long()
    return adj


def predict_case(data, args):
    data['m_d_matrix'] = make_adj(data['m_d'], (args.miRNA_number, args.disease_number))
    m_d_matrix = data['m_d_matrix']
    one_index = []
    zero_index = []
    for i in range(m_d_matrix.shape[0]):
        for j in range(m_d_matrix.shape[1]):
            if m_d_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)

    random.shuffle(one_index)
    one_index = np.array(one_index)
    random.shuffle(zero_index)
    zero_index = np.array(zero_index)

    train = np.concatenate(
        (one_index, zero_index[:int(args.negative_rate * len(one_index))]))
    mm = data['mm_f'] * np.where(data['mm_f'] == 0, 0, 1) + get_gaussian(data['m_d_matrix']) * np.where(
        data['mm_f'] == 1, 0, 1)
    dd = data['dd_s'] * np.where(data['dd_s'] == 0, 0, 1) + get_gaussian(data['m_d_matrix'].t()) * np.where(
        data['dd_s'] == 1, 0, 1)
    data['mm'] = {'data_matrix': mm, 'edges': get_edge_index(mm)}
    data['dd'] = {'data_matrix': dd, 'edges': get_edge_index(dd)}
    data['train'] = train


def data_processing(data, args):
    """
    Prepares all positive and negative samples and constructs the original feature matrices
    based on functional/semantic similarity and GIP kernel, as described in the paper.
    """
    md_adj = make_adj(data['md'], (args.miRNA_number, args.disease_number))
    data['md_adj'] = md_adj
    
    one_index = []
    zero_index = []
    for i in range(md_adj.shape[0]):
        for j in range(md_adj.shape[1]):
            if md_adj[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    data['pos_samples'] = np.array(one_index)
    data['neg_samples'] = np.array(zero_index)

    km = get_gaussian(md_adj)
    kd = get_gaussian(md_adj.t())

    ms_func = data['mf']
    ds_sem = data['dss']

    m_integrated_sim = np.where(ms_func != 0, ms_func, km)
    d_integrated_sim = np.where(ds_sem != 0, ds_sem, kd)

    data['ms'] = m_integrated_sim
    data['ds'] = d_integrated_sim


def k_matrix(matrix, k=20):
    """
    Build adjacency matrix for k-nearest neighbor graph
    """
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-matrix, axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k]] = matrix[i, idx_sort[i, :k]]
        knn_graph[idx_sort[i, :k], i] = matrix[idx_sort[i, :k], i]
    return knn_graph + np.eye(num)

def get_data(args):
    data = dict()
    
    mf = np.loadtxt(args.data_dir + 'miRNA functional similarity matrix.txt', dtype=np.float64)
    mfw = np.loadtxt(args.data_dir + 'miRNA functional similarity weight matrix.txt', dtype=np.float64)

    ds1 = np.loadtxt(args.data_dir + 'disease semantic similarity matrix 1.txt', dtype=np.float64)
    ds2 = np.loadtxt(args.data_dir + 'disease semantic similarity matrix 2.txt', dtype=np.float64)
    dsw = np.loadtxt(args.data_dir + 'disease semantic similarity weight matrix.txt', dtype=np.float64)

    if args.dd2 == True:
        dss = (ds1 + ds2) / 2
    else:
        dss = ds1

    args.miRNA_number = int(mf.shape[0])
    args.disease_number = int(dss.shape[0])

    data['mf'] = mf
    data['dss'] = dss
    data['mfw'] = mfw
    data['dsw'] = dsw
    data['d_num'] = np.loadtxt(args.data_dir + 'disease number.txt', delimiter='\t', dtype=str)[:, 1]
    data['m_num'] = np.loadtxt(args.data_dir + 'miRNA number.txt', delimiter='\t', dtype=str)[:, 1]
    data['md'] = np.loadtxt(args.data_dir + 'known disease-miRNA association number.txt', dtype=int) - 1
    return data, args


def get_gaussian(adj):
    Gaussian = np.zeros((adj.shape[0], adj.shape[0]), dtype=np.float64)
    gamaa = 1
    sumnorm = 0
    for i in range(adj.shape[0]):
        norm = np.linalg.norm(adj[i]) ** 2
        sumnorm = sumnorm + norm
    gama = gamaa / (sumnorm / adj.shape[0])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            Gaussian[i, j] = math.exp(-gama * (np.linalg.norm(adj[i] - adj[j]) ** 2))

    return Gaussian
