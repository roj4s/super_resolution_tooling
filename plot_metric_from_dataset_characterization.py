import pandas as pd
from matplotlib import pyplot as plt
import argparse

def plot_hist(dataset_addr, metric, log=False, title=None, xlabel=None):
    d = pd.read_csv(dataset_addr)
    print(f"Dataset shape: {d.shape}")
    if metric not in d.columns:
        raise f"Column {metric} not found"

    d[metric].hist()
    if xlabel is None:
        xlabel = metric.upper()

    plt.xlabel(xlabel)
    if title is not None:
        plt.title(title)

    if log:
        plt.yscale('log')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot dataset metrics')
    parser.add_argument('data_dir', type=str, help='dataset characterization' \
                        'csv file')
    parser.add_argument('--metric', type=str, help='Metric to plot, defaults to '\
                        'rmse', default='rmse')
    parser.add_argument('--log', action='store_true',
                    help='Apply log scale on y axis', default=False)
    parser.add_argument('--title', type=str, help='Optional chart title',
                        default=None)
    parser.add_argument('--xlabel', type=str, help='Optional optional x label',
                        default=None)


    args = parser.parse_args()
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    plot_hist(args.data_dir, args.metric, args.log, args.title,
              args.xlabel)
