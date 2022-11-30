from os import path
from utils import IMG_PATH, OUT_PATH

import pandas as pd
import matplotlib.pyplot as plt

colors = {
    'BLUE' : '#3264B0',
    'GREEN' : '#197D30',
    'RED' : '#AF3230',
    'ORANGE' : '#DD8120',
    'PURPLE' : '#490160',
    'SKY' : '#32BBAA',
    'GRAY' : '#6B6B6B',
}

def loss_plot(name, color=colors['ORANGE']):
    df = pd.read_csv(path.join(OUT_PATH, f'losses/loss_{name}.csv'), header=1)

    plt.plot(df.values, color = color, label=name)
    plt.xlabel(f'Epochs', fontweight = 'bold')
    plt.ylabel(f'Loss', fontweight = 'bold')
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(path.join(IMG_PATH, f'loss_{name}.jpg'), dpi=250)

if __name__ == '__main__':
    name = input('Name: ')
    loss_plot(name, colors['ORANGE'])