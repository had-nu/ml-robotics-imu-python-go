"""
Data loading utilities for UCI HAR dataset.
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../../data/raw/UCI-HAR Dataset')

def load_features_train():
    filepath = os.path.join(DATA_DIR, 'train', 'X_train.txt')

    df = pd.read_csv(filepath, sep=r'\s+', header=None)

    print(f"\nLoaded: {filepath}")
    print(f" Shape: {df.shape}")
    print(f" Primeiras 3 colunas, 2 linhas: ")
    print(df.iloc[:2, :3])

    return df


def load_labels_train():
    filepath = os.path.join(DATA_DIR, 'train', 'y_train.txt')

    df = pd.read_csv(filepath, header=None)

    print(f"\nLoaded: {filepath}")
    print(f" Shape: {df.shape}")
    print(f" Primeiros 5 labels: {df.head()}")

    return df


def load_features_test():
    pass


def load_labels_test():
    pass


def main():
    X_train = load_features_train()
    y_train = load_labels_train()

    print(f"\nTotal de features: {X_train.shape[1]}")
    print(f"Total de samples: {y_train.shape[0]}")


if __name__ == "__main__":
    main()
