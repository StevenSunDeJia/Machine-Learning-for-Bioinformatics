import pandas as pd
import os
import numpy as np

from data_file_utils import load_metadata, seq_name_to_clinical_name, extract_er_status

def prepare_network_genes(file_path = 'Kong_Paper/data/ppis/HomoSapiens_binary_hq.txt'):

    df = pd.read_csv(file_path, sep='\t')
    df = df.dropna(subset=['Uniprot_A', 'Uniprot_B', 'Gene_A', 'Gene_B'])
    df = df[['Gene_A', 'Gene_B']]
    network_genes = set(df['Gene_A']).union(set(df['Gene_B']))

    return df, network_genes

def prepare_seq_genes(dir = "Kong_Paper/data/patient_genes"):

    file_path = ""

    os.makedirs(dir, exist_ok=True)
    files = os.listdir(dir)
    for file in files:
        if file[-4:] == '.tsv':
            file_path = os.path.join(dir, file)
            break

    df = pd.read_csv(file_path, sep='\t', skiprows=1)
    df = df.dropna(subset=['gene_id', 'gene_name', 'gene_type', 'unstranded'])
    df = df[['gene_name', 'unstranded']]
    seq_genes = set(df['gene_name'])

    # save to seq_genes
    pd.DataFrame(seq_genes).to_csv('Kong_Paper/data/patient_genes/seq_genes.csv', index=False)
    print("seq_genes.csv created successfully.")

def compute_valid_genes(file_path = 'Kong_Paper/data/patient_genes/seq_genes.csv'):

    seq_genes = pd.read_csv(file_path, sep=',')
    seq_genes = set(seq_genes['0'])
    df, network_genes = prepare_network_genes()
    valid_genes = seq_genes.intersection(network_genes)

    # save to valid_genes
    pd.DataFrame(sorted(list(valid_genes))).to_csv('Kong_Paper/data/patient_genes/valid_genes.csv', index=False)
    print("valid_genes.csv created successfully.")

    file_path = 'Kong_Paper/data/patient_genes/valid_genes.csv'
    valid_genes = pd.read_csv(file_path, sep=',')
    valid_genes = list(valid_genes['0'])

    df = df[df['Gene_A'].isin(valid_genes) & df['Gene_B'].isin(valid_genes)]
    combined_unique_genes = set(df['Gene_A']).union(set(df['Gene_B']))
    missing_genes = set(valid_genes) - combined_unique_genes
    valid_genes = set(valid_genes) - missing_genes

    # update valid_genes
    pd.DataFrame(sorted(list(valid_genes))).to_csv('Kong_Paper/data/patient_genes/valid_genes.csv', index=False)
    print("valid_genes.csv updated successfully.")

    return df

def build_adjacency_matrix():

    prepare_seq_genes()
    df = compute_valid_genes()

    genes_a = df['Gene_A'].to_numpy()
    genes_b = df['Gene_B'].to_numpy()

    unique_genes = np.unique(np.concatenate((genes_a, genes_b)))
    gene_index = {gene: idx for idx, gene in enumerate(unique_genes)}
    size = len(unique_genes)
    adjacency_matrix = np.zeros((size, size), dtype=int)

    indices_a = np.vectorize(gene_index.get)(genes_a)
    indices_b = np.vectorize(gene_index.get)(genes_b)

    adjacency_matrix[indices_a, indices_b] = 1
    adjacency_matrix[indices_b, indices_a] = 1
    np.fill_diagonal(adjacency_matrix, 1)

    adj_df = pd.DataFrame(adjacency_matrix, index=unique_genes, columns=unique_genes)
    adj_df.to_csv('Kong_Paper/data/ppis/adjacency_matrix.csv', index=True)
    print("adjacency_matrix.csv created successfully.")

def get_gene_file_names(dir = "Kong_Paper/data/patient_genes"):

    os.makedirs(dir, exist_ok=True)
    files = os.listdir(dir)
    gene_files = []

    for file in files:
        if file[-4:] == '.tsv':
            gene_files.append(file)

    return gene_files, dir

def build_X_and_y(file_path = 'Kong_Paper/data/patient_genes/valid_genes.csv'): 

    gene_files, dir = get_gene_file_names()

    df = pd.read_csv(file_path, sep=',')
    valid_genes = list(df['0'])

    metadata = load_metadata()
    mapping = seq_name_to_clinical_name(metadata)

    sequences = []
    er_clinicals = []

    for i, seq_name in enumerate(gene_files):

        clinical_name = mapping[seq_name]
        clinical_name = os.path.join(dir, clinical_name)

        try:
            print(i+1, "File parsed successfully.")
            clinical_status = extract_er_status(clinical_name)
            er_clinicals.append(clinical_status)

            file_path = os.path.join(dir, seq_name)
            df = pd.read_csv(file_path, sep='\t', skiprows=1)
            df = df.dropna(subset=['gene_id', 'gene_name', 'gene_type', 'unstranded'])
            df = df[['gene_name', 'unstranded']].sort_values(by='gene_name')
            df = df[df['gene_name'].isin(valid_genes)]
            df = df.drop_duplicates(subset='gene_name', keep='first')

            sequences.append(df['unstranded'].to_numpy())
         
        except Exception as e:
            print(i+1, e)
            continue

    X = np.vstack(sequences)

    # Convert to a list of 1s and 0s
    er_clinicals = [1 if er_clinical == 'Positive' else 0 for er_clinical in er_clinicals]
    y = np.array(er_clinicals)

    df = pd.DataFrame(X)
    df.to_csv("Kong_Paper/data/patient_genes/X.csv", index=False, header=False)
    print("X.csv created successfully.")

    df = pd.DataFrame(y)
    df.to_csv("Kong_Paper/data/patient_genes/y.csv", index=False, header=False)
    print("y.csv created successfully.")