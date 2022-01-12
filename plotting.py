import matplotlib as mpl
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

# These are global matplotlib settings to emphasize uniformity across figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
width = 15
height = width/1.618 # Use the golden ratio.
mpl.rc('figure', figsize=(width,height)) 
plt.style.use('ggplot') # This plot style is borrowed from R's ggplot2.

# Now we want to specify where to save figures.
PROJECT_ROOT_DIR = '.' #the current working directory
IMAGES_PATH = Path(PROJECT_ROOT_DIR)/'figures'
IMAGES_PATH.mkdir(exist_ok=True, parents=True)

def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    '''
    Save the current matplotlib figure to a specified global path.
    '''
    fname = fig_id+'.'+fig_extension
    path = IMAGES_PATH/fname
    print(f'saving figure: {fig_id}')
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def prepare_data_for_plotting(total_rewards_history, smoothing_window=5):
    df = pd.DataFrame(total_rewards_history, columns=['Total Reward'])
    df[f'Total Reward Rolling Mean (k={smoothing_window})'] = df['Total Reward'].rolling(smoothing_window).mean()
    return df

def plot_history(total_rewards_history, name=None):
    df = prepare_data_for_plotting(total_rewards_history)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(df, '.-')
    ax.set_title('Snake Agent Learning Curve')
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Total Episode Reward')
    plt.legend(df.columns.to_list(), loc='best')
    fig_id = f'snake-agent-learning-curve-{name}' if name else 'snake-agent-learning-curve'
    save_fig(fig_id)
    plt.show()
    plt.close()