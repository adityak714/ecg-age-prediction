import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='plot learning curve.')
    parser.add_argument('history_file', type=str,
                        help="path to history file.")
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('--save', default='',
                        help='save the plot in the given file')
    args = parser.parse_args()

    if args.plot_style:
        plt.style.use(args.plot_style)

    df = pd.read_csv(args.history_file)

    # Plot MAE
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(df['train_rmse'])), df['train_rmse'], label='train_rmse', color='blue')
    ax.scatter(np.arange(len(df['train_rmse'])), df['train_mae'], label='train_mae', color='blue', marker='.')
    ax.plot(np.arange(len(df['train_rmse'])), df['valid_rmse'], label='valid_rmse', color='orange')
    ax.scatter(np.arange(len(df['train_rmse'])), df['valid_mae'], label='valid_mae', color='orange', marker='.')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MAE (years)')
    ax.legend()
    #ax.set_ylim((4, 24))
    axt = ax.twinx()

    # Plot learning rate
    axt.step(np.arange(len(df['train_rmse'])), df['lr'], label='train', alpha=0.4, color='k')
    axt.set_yscale('log')
    axt.set_ylabel('learning rate', alpha=0.4, color='k')
    axt.set_ylim((1e-11, 1e-4))

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()
