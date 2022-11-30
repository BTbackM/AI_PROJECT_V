import math
import pandas as pd
import utils
import torch
import torchvision.transforms as transforms

from os import path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split
from utils import DATA_PATH

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences['SEQUENCE'].values
        self.sc = MinMaxScaler()
        # self.labels = self.sc.fit_transform(sequences['TM'].values.reshape(-1, 1)).reshape(-1)
        self.labels = sequences['TM'].values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = [ord(char) for char in self.sequences[idx]]
        sequence = torch.FloatTensor(sequence)
        label = self.labels[idx]
        label = label

        return sequence, label

class SequenceLabeledDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences['SEQUENCE'].values
        self.labels = sequences['LABEL'].values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = [ord(char) for char in self.sequences[idx]]
        sequence = torch.FloatTensor(sequence)
        label = self.labels[idx]
        label = label

        return sequence, label

def load_dataloader():
    sequences = pd.read_csv(path.join(DATA_PATH, 'sequences_labeled.csv'), sep='\t')

    dataset = SequenceLabeledDataset(
        sequences,
    )

    train_set, test_set = random_split(dataset, [math.floor(len(dataset) * 0.7), math.ceil(len(dataset) * 0.3)])

    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    return train_loader, test_loader

# NOTE: Load data

def load_data():
    dataset = pd.read_csv(path.join(DATA_PATH, 'sequences.csv'), sep='\t')

    sequences = dataset['SEQUENCE'].values
    labels = dataset['TM'].values

    sequences = [[ord(char) for char in sequence] for sequence in sequences]
    sequences = [torch.FloatTensor(sequence) for sequence in sequences]
    labels = torch.FloatTensor(labels)

    X_train, X_test, Y_train, Y_test = train_test_split(sequences, labels, test_size=0.3, random_state=42)

    return X_train, X_test, Y_train, Y_test