from data import load_data, load_dataloader
from graphics import loss_plot
from lstm import LSTM, LSTM_WP, LSTM_P, train, predict
from os import path, listdir, makedirs
from utils import timeit
from utils import OUT_PATH

import pandas as pd
import torch
import torch.nn as nn

# NOTE: Verify cuda availability

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# NOTE: Load data

train_loader, test_loader = load_dataloader()

# NOTE: Build model

params = {
    'input_size': 1,
    'hidden_size': 32,
    'num_layers': 1,
    'num_classes' : 2,
}

# NOTE: Model persistency path

name = f'lstm_labeled_10'
model_path = path.join(OUT_PATH, 'model')
if not path.exists(model_path):
    makedirs(model_path)
model_path = path.join(model_path, f'model_{name}.pt')

if path.exists(model_path):
    print('Model found')
    model = torch.load(model_path)
else:
    print('Model not found')
    # NOTE: Declare model

    model = LSTM(**params).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, 10, criterion, optimizer, device, name)

    loss_plot(name)

    # NOTE: Save model
    torch.save(model, model_path)

model.eval()
predict(model, test_loader, device)