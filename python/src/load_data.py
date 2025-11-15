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

    """
    print(f"\nLoaded: {filepath}")
    print(f" Shape: {df.shape}")
    print(" Primeiras 3 colunas, 2 linhas: ")
    print(df.iloc[:2, :3])
    """

    return df


def load_labels_train():
    filepath = os.path.join(DATA_DIR, 'train', 'y_train.txt')

    df = pd.read_csv(filepath, header=None)

    """
    print(f"\nLoaded: {filepath}")
    print(f" Shape: {df.shape}")
    print(f" Primeiros 5 labels: {df.head()}")
    """

    return df


def load_features_test():
    filepath = os.path.join(DATA_DIR, 'test', 'X_test.txt')

    df = pd.read_csv(filepath, sep=r'\s+', header=None)

    """
    print(f"\nLoaded: {filepath}")
    print(f"Shape: {df.shape}")
    print(" Primeiras 3 colunas, 2 linhas: ")
    print(df.iloc[:2, :3])
    """

    return df


def load_labels_test():
    filepath = os.path.join(DATA_DIR, 'test', 'y_test.txt')

    df = pd.read_csv(filepath, header=None)

    """
    print(f"\nLoaded: {filepath}")
    print(f" Shape: {df.shape}")
    print(f" Primeiros 5 labels: {df.head()}")
    """

    return df


def load_feature_names():
    filepath = os.path.join(DATA_DIR, 'features.txt')
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['id', 'name'])
    
    return df['name'].values


def load_activity_labels():
    filepath = os.path.join(DATA_DIR, 'activity_labels.txt')
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['id', 'activity'])
    activity_map = dict(zip(df['id'], df['activity']))

    print(f"\nLoaded {len(activity_map)} activity labels")
    
    return activity_map
 

def main():
    X_train = load_features_train()
    y_train = load_labels_train()
    X_test = load_features_test()
    y_test = load_labels_test()
    
    names = load_feature_names()
    X_train.columns = names
    X_test.columns = names
    
    y_train.columns = ['activity_id']
    y_test.columns = ['activity_id']

    activity_map = load_activity_labels()
    y_train['activity_name'] = y_train['activity_id'].map(activity_map)
    y_test['activity_name'] = y_test['activity_id'].map(activity_map)
    
    print(f"\nDataset completo:")
    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"\nPrimeiras 3 features: {list(X_train.columns[:3])}")
    
    print(f"\nVerificação - X_train com nomes:")
    print(X_train.iloc[:2, :3])

    print(f"\nVerificação - y_train com nomes:")
    print(y_train.head())


if __name__ == "__main__":
    main()
