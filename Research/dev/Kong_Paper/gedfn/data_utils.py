import numpy as np
import networkx as nx
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

def generate_scale_free_graph(p, m=1):
    """
    Generate a scale-free graph using the Barabási–Albert model.
    
    Args:
        p (int): Number of nodes (features).
        m (int): Number of edges to attach from a new node to existing nodes.
    
    Returns:
        G (networkx.Graph): A scale-free network.
    """
    G = nx.barabasi_albert_graph(p, m, seed=42)
    return G

def compute_distance_matrix(G):
    """
    Compute the pairwise shortest-path distance matrix D for graph G.
    
    Args:
        G (networkx.Graph): The feature graph.
    
    Returns:
        D (np.ndarray): A (p x p) matrix of shortest-path distances.
    """
    p = G.number_of_nodes()
    D = np.zeros((p, p))
    for i in range(p):
        lengths = nx.single_source_shortest_path_length(G, i)
        for j in range(p):
            D[i, j] = lengths.get(j, np.inf)
    return D

def generate_covariance_matrix(D, decay=0.7):
    """
    Generate a covariance matrix R from the distance matrix D.
    
    Args:
        D (np.ndarray): Distance matrix.
        decay (float): Decay factor; R[i,j] = decay^(D[i,j]).
    
    Returns:
        R (np.ndarray): Covariance matrix.
    """
    R = np.power(decay, D)
    np.fill_diagonal(R, 1.0)
    return R

def select_true_predictors(G, p0, s0=0.0):
    """
    Select true predictors based on the feature graph.
    
    The method selects nodes with high degree (cores) and includes some of their neighbors.
    
    Args:
        G (networkx.Graph): Feature graph.
        p0 (int): Total number of true predictors to select.
        s0 (float): Proportion of 'singletons' amongst the true predictors.
    
    Returns:
        true_idx (np.ndarray): Array of indices for true predictors.
    """
    degrees = dict(G.degree())
    # Sort nodes by degree (highest first)
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    
    # Choose cores: for example, choose k = max(1, p0//20) top nodes
    k = max(1, p0 // 20)
    cores = [node for node, deg in sorted_nodes[:k]]

    # Determine the number of singletons
    nb_singletons = int(np.ceil(s0 * p0))

    # Remove singletons from the total number of predictors
    p0 -= nb_singletons
    
    true_predictors = set(cores)
    # Add neighbors of each core until we have p0 predictors
    for core in cores:
        neighbors = list(G.neighbors(core))
        np.random.shuffle(neighbors)
        for neighbor in neighbors:
            if len(true_predictors) < p0:
                true_predictors.add(neighbor)
            else:
                break
        if len(true_predictors) >= p0:
            break
    # If not enough, add additional high-degree nodes
    for node, deg in sorted_nodes:
        if len(true_predictors) < p0:
            true_predictors.add(node)
        else:
            break

    while nb_singletons > 0:
        singleton = np.random.choice(list(G.nodes()))
        if singleton not in true_predictors:
            true_predictors.add(singleton)
            nb_singletons -= 1

    true_idx = np.array(list(true_predictors))[:p0]
    return true_idx

def generate_outcome(X, true_idx, b_range=(0.1, 0.2), threshold=0.5, link='sigmoid'):
    """
    Generate binary outcome variables using a generalized linear model.
    
    Args:
        X (np.ndarray): Input data matrix (n x p).
        true_idx (np.ndarray): Indices of true predictors.
        b_range (tuple): Range for sampling coefficients.
        threshold (float): Threshold for converting probabilities to binary outcomes.
        link (str): Link function ('sigmoid' or 'tanh_quad').
        
    Returns:
        y (np.ndarray): Binary outcome vector (n,).
        b (np.ndarray): Coefficients for true predictors.
        b0 (float): Intercept.
        prob (np.ndarray): Computed probabilities.
    """
    p0 = len(true_idx)
    # Sample coefficients uniformly from b_range
    b = np.random.uniform(b_range[0], b_range[1], size=p0)
    # Randomly flip signs (to allow both positive and negative effects)
    signs = np.random.choice([-1, 1], size=p0)
    b = b * signs
    b0 = np.random.uniform(b_range[0], b_range[1])
    
    # Compute linear combination for each sample
    linear_term = np.dot(X[:, true_idx], b) + b0
    
    if link == 'sigmoid':
        prob = 1 / (1 + np.exp(linear_term)) # MT: was np.exp(-linear_term)
    elif link == 'tanh_quad':
        # Example non-monotone function: weighted tanh plus quadratic, then min-max scaling.
        raw = 0.7 * np.tanh(linear_term) + 0.3 * (linear_term ** 2)
        raw_min, raw_max = raw.min(), raw.max()
        if raw_max > raw_min:
            prob = (raw - raw_min) / (raw_max - raw_min)
        else:
            prob = np.zeros_like(raw)
    else:
        raise ValueError("Unknown link function.")
    
    # Generate binary outcomes by thresholding the probabilities
    y = (prob > threshold).astype(int)
    return y, b, b0, prob

def generate_synthetic_data(n=400, p=5000, p0=40, s0=0.0, decay=0.7, m=1, link='sigmoid',
                            b_range=(0.1, 0.2), threshold=0.5, random_seed=42):
    """
    Generate synthetic gene expression data and binary outcomes as described in Kong & Yu (2018).
    
    Args:
        n (int): Number of samples.
        p (int): Number of features (genes).
        p0 (int): Number of true predictors.
        decay (float): Decay factor for covariance matrix.
        m (int): Number of edges to attach from a new node in the Barabási–Albert model.
        link (str): Link function ('sigmoid' or 'tanh_quad').
        b_range (tuple): Range for sampling coefficients.
        threshold (float): Threshold for binary outcome generation.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        X (np.ndarray): Generated input data (n x p).
        y (np.ndarray): Binary outcomes (n,).
        true_idx (np.ndarray): Indices of true predictors.
        R (np.ndarray): Covariance matrix used.
        G (networkx.Graph): The generated feature graph.
        b (np.ndarray): True coefficients for the predictors.
        b0 (float): Intercept.
        prob (np.ndarray): Underlying probabilities.
    """
    np.random.seed(random_seed)
    
    # Generate a scale-free feature graph
    G = generate_scale_free_graph(p, m=m)
    
    # Compute the distance matrix D based on shortest paths in G
    D = compute_distance_matrix(G)
    
    # Generate covariance matrix R: R[i,j] = decay^(D[i,j])
    R = generate_covariance_matrix(D, decay=decay)
    
    # Generate n samples from a multivariate normal with covariance R
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=R, size=n)
    
    # Select true predictors from the graph (aiming for clique-like structures)
    true_idx = select_true_predictors(G, p0, s0)
    
    # Generate binary outcomes using a generalized linear model
    y, b, b0, prob = generate_outcome(X, true_idx, b_range=b_range, threshold=threshold, link=link)
    
    return X, y, true_idx, R, G, b, b0, prob