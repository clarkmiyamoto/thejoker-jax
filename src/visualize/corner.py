import numpy as np
import seaborn as sns
sns.set_theme(style="dark")
import pandas as pd

def corner(data: np.ndarray,
           labels: list[str] = None,
           show_titles: bool = False,
           bins: int=20,
           levels: list[float]=[0.118, 0.393, 0.675, 0.864],
           ):
    '''
    Args:
        samples: Shape (num_samples, num_params)
        labels: List of names of parameters. Shape (num_params,)
        show_titles: Whether to show the maximum of the KDE for each parameter. Default is False.
        bins: Number of bins for the histogram. Default is 20.
        levels: Confidence levels to use to draw contours.
                Default is [0.118, 0.393, 0.675, 0.864], same as DFM's `corner.py`.
    '''
    # SNS uses pandas under the hood
    df = pd.DataFrame(data, columns=labels)

    # Plot the data
    g = sns.PairGrid(df, corner=True, height=4.5, )

    g.map_lower(sns.scatterplot, s=5, color='.15',)
    g.map_lower(sns.histplot, bins=bins, pthresh=.1, cmap='mako')
    g.map_lower(sns.kdeplot, levels=levels, color='w', linewidths=1,)
    g.map_diag(sns.kdeplot)

    ### Bells and whistles
    # Get argmax using diagonal plots   
    if show_titles:
        maximum_param_estimator = []
        for i in range(len(df.columns)):
            x = g.diag_axes[i].lines[0].get_xdata()
            y = g.diag_axes[i].lines[0].get_ydata()
            argmax = x[np.argmax(y)]
            maximum_param_estimator.append(argmax)


    for i, row_param in enumerate(df.columns):
        for j, _ in enumerate(df.columns):

            # Diagonal: Histogram of marginals
            if i == j:
                ax = g.axes[i, j]
                
                ### Below add options:
                if show_titles:
                    ax.axvline(maximum_param_estimator[i], color="red", lw=1.5, zorder=5, linestyle="--")
                    ax.set_title(f"{row_param} = {maximum_param_estimator[i]:.2f}")

            # Lower-triangle scatter: Cross section of two-point distributions
            elif i > j:
                ax = g.axes[i, j]

                ### Below add options:
                if show_titles:
                    ax.axvline(maximum_param_estimator[j], color="red", lw=1.0, zorder=5, linestyle="--")
                    ax.axhline(maximum_param_estimator[i], color="red", lw=1.0, zorder=5, linestyle="--")