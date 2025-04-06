
from gedfn_keras import create_baseline_model, create_graph_model
from data_utils import generate_synthetic_data

import numpy as np
import pandas as pd

import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

## Returns a function that creates a baseline model

def build_compile_baseline_model(input_dim, hidden_dims):

    def builder():
        model = create_baseline_model(input_dim=input_dim, hidden_dims=hidden_dims)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    return builder

## Returns a function that creates a graph-embedded model

def build_compile_graph_model(input_dim, hidden_dims, A):

    def builder():
        model = create_graph_model(input_dim, hidden_dims, A)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    return builder

## Fixed Dataset, Multiple Runs

def evaluate_keras_model(build_compile_model, X_train, y_train, X_val, y_val, X_test, y_test, 
                         n_runs=10, epochs=100, batch_size=32, seed_start=0):
    """
    Evaluates a Keras model over n_runs by resetting the random seed,
    creating, compiling, and training the model for each run. The function
    collects classification metrics from the test set and aggregates them.

    Parameters:
      build_compile_model: A function that creates and compiles a Keras model.
      X_train, y_train: Training data and labels.
      X_val, y_val: Validation data and labels.
      X_test, y_test: Test data and labels.
      n_runs: Number of evaluation runs.
      epochs: Number of epochs to train the model each run.
      batch_size: Batch size for training.
      seed_start: The starting value for the random seed; an offset is added for each run.
    
    Returns:
      A DataFrame summarizing the mean and standard deviation of each classification metric.
    """
    
    def flatten_report(report):
        # Flatten the nested classification report dictionary.
        flat = {}
        for key, value in report.items():
            if isinstance(value, dict):
                for metric, score in value.items():
                    flat[f"{key}_{metric}"] = score
            else:
                flat[key] = value
        return flat

    # Define callbacks for early stopping.
    callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    reports = []
    
    for i in range(n_runs):

        # Set a unique random seed for reproducibility and variability.
        seed_val = seed_start + i
        keras.utils.set_random_seed(seed_val)
        
        # Create and compile a new instance of the model.
        model = build_compile_model()
        
        # Train the model.
        model.fit(X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
        
        # Generate predictions on the test data.
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
        
        # Create a classification report and flatten its nested structure.
        report = classification_report(y_test, y_pred, output_dict=True)
        reports.append(flatten_report(report))
    
    # Convert collected reports to a DataFrame.
    df_reports = pd.DataFrame(reports)
    
    # Compute mean and standard deviation for each metric.
    mean_df = df_reports.mean()
    std_df = df_reports.std()
    
    # Combine into one summary DataFrame.
    summary_df = pd.DataFrame({"mean": mean_df, "std": std_df})
    
    print("Summary Classification Report (mean ± std over {} runs):".format(n_runs))
    print(summary_df)
    
    return summary_df

## Multiple Datasets, Multiple Runs

def run_synthetic():

    # Parameters (for demonstration, p can be set lower than 5000)

    n = 400      # number of samples
    p = 5000     # number of features (adjust to 5000 for full-scale simulation)
    p0 = 40      # number of true predictors
    s0 = 0.0     # proportion of singletons (0.0, 0.5, 1.0)

    hidden_dims = [64, 16]
    n_datasets = 10
    n_runs = 10

    for seed in range(n_datasets):

        print("Create Dataset", seed)

        # Generate the synthetic data
        X, y, true_idx, R, G, b, b0, prob = generate_synthetic_data(n=n, p=p, p0=p0, s0=s0, random_seed=seed)

        print(f"n={n}, number of positive {np.sum(y)}")

        A = nx.adjacency_matrix(G).todense()
        np.fill_diagonal(A, 1)

        # Split the data: 80% train, 10% validation, 10% test.

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_temp = scaler.transform(X_temp)

        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=seed)


        print()
        print("Baseline Model")

        evaluate_keras_model(build_compile_baseline_model(p, hidden_dims), X_train, y_train, X_val, y_val, X_test, y_test, n_runs=n_runs)

        print()
        print("Graph-Embedded Model:")

        evaluate_keras_model(build_compile_graph_model(p, hidden_dims, A), X_train, y_train, X_val, y_val, X_test, y_test, n_runs=n_runs)

        print()

