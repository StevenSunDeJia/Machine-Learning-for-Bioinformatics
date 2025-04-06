import pandas as pd
import numpy as np

from transform_real_data import build_X_and_y, build_adjacency_matrix
from run_0001_gedfn_vs_baseline_10x10 import build_compile_baseline_model, build_compile_graph_model, evaluate_keras_model

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import time

def prepare_adj_matrix(file_path = 'Kong_Paper/data/ppis/adjacency_matrix.csv'):

    df = pd.read_csv(file_path, sep=',')
    adj_matrix = df.to_numpy()
    adj_matrix = adj_matrix[:, 1:].astype(np.float32)

    print("Adjancency matrix prepared successfully.")
    return adj_matrix, adj_matrix.shape[0]

def prepare_X_and_y(file_path_X = 'Kong_Paper/data/patient_genes/X.csv', file_path_y = 'Kong_Paper/data/patient_genes/y.csv'):

    df = pd.read_csv(file_path_X, sep=',')
    X = df.to_numpy().astype(np.float32)
    df = pd.read_csv(file_path_y, sep=',')
    y = df.to_numpy().astype(np.float32)

    print("X and y prepared successfully.")
    return X, y

def run(n_runs=1):

    X, y = prepare_X_and_y()
    A, p = prepare_adj_matrix()

    hidden_dims = [64, 16]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_temp = scaler.transform(X_temp)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

    print()
    print("Baseline Model")

    start_time = time.time()
    evaluate_keras_model(build_compile_baseline_model(p, hidden_dims), X_train, y_train, X_val, y_val, X_test, y_test, n_runs=n_runs)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Baseline Model executed in: {duration:.6f} seconds")

    print()
    print("Graph-Embedded Model:")

    start_time = time.time()
    evaluate_keras_model(build_compile_graph_model(p, hidden_dims, A), X_train, y_train, X_val, y_val, X_test, y_test, n_runs=n_runs)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Graph-Embedded Model executed in: {duration:.6f} seconds")

    print()

def main():

    """ build_adjacency_matrix()

    build_X_and_y() """

    print("Running main()")

    run()

if __name__ == "__main__":
    main()
