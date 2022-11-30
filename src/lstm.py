import math
import pandas as pd
import torch
import torch.nn as nn

from os import path
from utils import timeit
from utils import OUT_PATH
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM_P(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(LSTM_P, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, bidirectional=False, batch_first=True)
        self.drop_out = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        # NOTE: Windowed LSTM
        x = x.reshape(-1, 1)
        x = list(torch.split(x, 32))

        lenghts = [len(xx) for xx in x]
        x = pad_sequence(x, batch_first=True, padding_value=0)
        xpadded = pack_padded_sequence(x, lenghts, batch_first=True, enforce_sorted=False)

        out, _ = self.lstm(xpadded)

        out, _ = pad_packed_sequence(out, batch_first=True, padding_value=0)
        y = out[-1, -1, :]

        # y = self.fc1(y)
        # y = self.relu(y)
        # y = self.bn1(y)
        # y = self.drop_out(y)
        y = self.fc_out(y)
        return y

class LSTM_WP(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, num_classes=1):
        super(LSTM_WP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, bidirectional=False, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # NOTE: Windowed LSTM
        x = x.reshape(-1, 1)
        x = list(torch.split(x, self.hidden_size))

        outputs = []
        for xx in x:
            out, _ = self.lstm(xx.unsqueeze(0))
            out = out[-1, -1, :]

            outputs.append(out.squeeze(0))
        outputs = torch.stack(outputs)

        y = self.fc_out(outputs)
        return y

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, num_classes=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, bidirectional=False, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # NOTE: Windowed LSTM
        x = x.reshape(-1, 1)
        
        out, (h_out, _) = self.lstm(x.unsqueeze(0))
        out = out[-1, -1, :]
        # h_out = h_out.view(-1, self.hidden_size)

        y = self.fc_out(out)
        return y

# NOTE: Train CNN model
def train(model, dataset, epochs, criterion, optimizer, device, name):
    losses = []
    dataset_size = len(dataset)

    loss = 0
    for epoch in range(epochs):
        for i, (sequences, labels) in enumerate(dataset):
            sequences = sequences.to(device)
            labels = labels.to(device)

            # NOTE: Forward pass
            outputs = model(sequences)
            loss = criterion(outputs.unsqueeze(0), labels)
            losses.append(loss.item())

            # NOTE: Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print(f'Epoch [{epoch + 1} / {epochs}], Loss: {loss.item():.4f}')
        
    # NOTE: Save loss history
    file = path.join(OUT_PATH, f'losses/loss_{name}.csv')
    try:
        losses_df = pd.DataFrame(losses, columns = ['loss'])
        losses_df.to_csv(file, index = False)
    except:
        print(f'Cannot save losses to {file}')

def predict(model, dataset, device):
    correct, total = 0, 0

    with torch.no_grad():
        for i, (sequences, labels) in enumerate(dataset):
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            # predicted = outputs.data
            _, predicted = torch.max(outputs.data, 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # correct += (abs(predicted - labels) < 5).sum().item()

        print(f'Accuracy of the model on the {total} test sequences: {100 * correct / total}%')