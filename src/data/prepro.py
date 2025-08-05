import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TensorSequenceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def normalize_data(X, Y):
    """
    Normalisiert X und Y mit ihren jeweils eigenen Statistiken.
    """
    # 1. Statistiken für X berechnen und anwenden
    # Berechne mean/std über die Batch- und Zeit-Dimensionen
    mean_X = X.mean(dim=(0, 1), keepdim=True)
    std_X = X.std(dim=(0, 1), keepdim=True)
    # Sicherheitscheck, um Division durch Null zu vermeiden
    std_X[std_X == 0] = 1.0
    X_norm = (X - mean_X) / std_X

    # 2. Statistiken für Y getrennt berechnen und anwenden
    mean_Y = Y.mean(dim=(0, 1), keepdim=True)
    std_Y = Y.std(dim=(0, 1), keepdim=True)
    # Sicherheitscheck
    std_Y[std_Y == 0] = 1.0
    Y_norm = (Y - mean_Y) / std_Y

    # Gib die normalisierten Daten und BEIDE Statistik-Paare zurück
    # Du brauchst sie später, um die Vorhersagen deines Modells zu de-normalisieren
    stats_X = (mean_X, std_X)
    stats_Y = (mean_Y, std_Y)
    
    return X_norm, Y_norm, stats_X, stats_Y