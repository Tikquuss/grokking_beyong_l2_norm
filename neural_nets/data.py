import torch
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset

import math

########################################################################################
########################################################################################

def create_single_data_loader(X, y, batch_size=None, verbose=True, **kwarg):
    dataset = TensorDataset(X, y)
    N = len(X)
    batch_size = N if batch_size is None else min(batch_size, N)
    dataloader = DataLoader(dataset, batch_size=batch_size, **kwarg)
    if verbose : print(f"data size = {N}, loader size = {len(dataloader)}")
    return dataloader

########################################################################################
########################################################################################

def create_data_loader(X_train, y_train, X_test, y_test, batch_size=None, eval_batch_size=None, verbose=True):
    N_train, N_test = len(X_train), len(X_test)
    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    batch_size = N_train if batch_size is None else min(batch_size, N_train)
    eval_batch_size_train = N_train if eval_batch_size is None else min(eval_batch_size, N_train)
    eval_batch_size_test = N_test if eval_batch_size is None else min(eval_batch_size, N_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    train_loader_for_eval = DataLoader(train_set, batch_size=eval_batch_size_train, shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size_test, shuffle=False, drop_last=False, num_workers=0)

    if verbose : 
        print(f"Data size : train = {N_train}, test = {N_test}")
        print(f"Loader size : train = {len(train_loader)}, train for val = {len(train_loader_for_eval)}, test = {len(test_loader)}")
    return train_loader, train_loader_for_eval, test_loader

########################################################################################
########################################################################################

def custom_train_test_split(X, y, r_train=0.4, random_state=None, train_class_proportions=None):
    if random_state is not None:
        np_state = np.random.get_state()
        np.random.seed(random_state)
    
    # Determine the unique classes and their counts
    classes, counts = np.unique(y, return_counts=True)
    C = len(classes)
    N = len(y)
    
    # Find the nearest multiple of C for N_train
    target_train_size = round(N * r_train)
    N_train = (target_train_size // C) * C  # Find the largest multiple of C less than or equal to target
    if N_train == 0:
        N_train = C  # Ensure at least one sample per class if r_train is very small
    
    # Default to uniform distribution if no specific class proportions are provided
    if train_class_proportions is None:
        train_class_proportions = [1 / C] * C
    else:
        # Normalize provided class proportions to ensure they sum to 1
        train_class_proportions = np.array(train_class_proportions) / sum(train_class_proportions)
    
    # Calculate the number of training samples per class based on the specified proportions
    train_counts = (N_train * train_class_proportions).astype(int)

    # Split indices by class
    train_indices = []
    test_indices = []

    for c, train_count in zip(classes, train_counts):
        cls_indices = np.where(y == c)[0]
        np.random.shuffle(cls_indices)
        
        # Use as many samples as possible if a class has fewer samples than needed
        train_indices.extend(cls_indices[:min(train_count, len(cls_indices))])
        test_indices.extend(cls_indices[min(train_count, len(cls_indices)):])

    # Extract training and test samples based on the indices
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    if random_state is not None:
        np.random.set_state(np_state)
        
    return X_train, X_test, y_train, y_test

########################################################################################
########################################################################################

def perfect_balance_train_test_split(X, y, r_train:float, random_state=None):
    if random_state is not None:
        np_state = np.random.get_state()
        np.random.seed(random_state)
    
    # Determine the unique classes and their counts
    classes, counts = np.unique(y, return_counts=True)
    C = len(classes)
    N = len(y)
    
    # Find the nearest multiple of C for N_train
    target_train_size = round(N * r_train)
    N_train = (target_train_size // C) * C  # Find the largest multiple of C less than or equal to target
    if N_train == 0:
        N_train = C  # Ensure at least one sample per class if r_train is very small
      
    # Split indices by class
    train_indices = []
    test_indices = []
    
    for c in classes:
        cls_indices = np.where(y == c)[0]
        np.random.shuffle(cls_indices)
        
        # Determine the number of training samples per class
        num_train_samples_per_class = N_train // C
        
        train_indices.extend(cls_indices[:min(num_train_samples_per_class, len(cls_indices))])
        test_indices.extend(cls_indices[min(num_train_samples_per_class, len(cls_indices)):])
    
    # Extract training and test samples based on the indices
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    if random_state is not None:
        np.random.set_state(np_state)

    return X_train, X_test, y_train, y_test

########################################################################################
########################################################################################

def split_and_create_data_loader(
    X, y, r_train:float, batch_size=None, eval_batch_size=None, 
    random_state=42, stratify=None, 
    balance=False, train_class_proportions=None,
    verbose=True):
    assert 0 < r_train < 1
    if train_class_proportions :
        X_train, X_test, y_train, y_test = custom_train_test_split(X, y, r_train=r_train, random_state=random_state, train_class_proportions=train_class_proportions)
    elif balance:
        X_train, X_test, y_train, y_test = perfect_balance_train_test_split(X, y, r_train=r_train, random_state=random_state)
    else :
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=r_train, random_state=random_state, stratify=stratify)

    # if verbose : 
    #     print(f"Data size : train = {len(X_train)}, test = {len(X_test)}")
    
    return create_data_loader(X_train, y_train, X_test, y_test, batch_size, eval_batch_size, verbose=verbose)

########################################################################################
########################################################################################

if __name__ == "__main__":

    d, C = 5, 3
    N = 10**2
    X = torch.randn(size=(N, d)) # (N, d)
    y = torch.randn(size=(N, C)) # (N, C)

    r_train=0.9
    data_seed=42 * 1

    batch_size=2**4
    eval_batch_size=2**4

    train_loader, train_loader_for_eval, test_loader = split_and_create_data_loader(
        X, y, r_train=r_train, batch_size=batch_size, eval_batch_size=eval_batch_size, random_state=data_seed, balance=False)

    for batch_x, batch_y in train_loader:
        print(batch_x.size(), batch_y.size())
        break