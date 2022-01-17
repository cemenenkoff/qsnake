import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import json

# global matplotlib settings
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
WIDTH = 15
HEIGHT = WIDTH/1.618
mpl.rc('figure', figsize=(WIDTH,HEIGHT))
plt.style.use('ggplot') # This plot style is borrowed from R's ggplot2.

def save_fig(outpath, tight_layout=True, resolution=300):
    '''
    saves the current matplotlib figure to a specified global path
    '''
    print(f'saving figure: {outpath}')
    if tight_layout:
        plt.tight_layout()
    plt.savefig(outpath, dpi=resolution)

def prepare_data_for_plotting(history, window_len=5):
    '''
    puts the historical total rewards data into a dataframe
    '''
    df = pd.DataFrame(history, columns=['Total Reward'])
    df[f'Total Reward Rolling Mean (k={window_len})'] =\
        df['Total Reward'].rolling(window_len).mean()
    return df

def plot_history(history, outpath=None, params=None):
    '''
    plots the historical total rewards data
    '''
    window_len = max(int(params.get('num_episodes')/5), 5)
    df = prepare_data_for_plotting(history, window_len=window_len)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(df, '.-')
    ax.set_title('Snake Agent Learning Curve')
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Total Episode Reward')
    if params:
        plt.text(0.5, -0.1, json.dumps(params), ha='center',
                 va='baseline', transform = ax.transAxes, size='small')
    plt.legend(df.columns.to_list(), loc='best')
    outpath = outpath if outpath else 'learning-curve.png'
    save_fig(outpath)
    plt.close('all')