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

    print(f"☑ Loaded {filepath}")
    print(f" Shape: {df.shape}")
    print(f" Primeiras 3 colunas, 2 linhas: ")
    print(df.iloc[:2, :3])

    return df


def main():
    """Testa o carregamento"""
    X_train = load_features_train()
    print(f"\n Total de features: {X_train.shape[1]}")


if __name__ == "__main__":
    main()
