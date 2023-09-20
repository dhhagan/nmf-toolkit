import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def cv_figure(data, **kwargs):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    """
    fig, ax = plt.subplots(1, figsize=(8, 6))
    
    # Plot
    ax.plot(data["MSE"]["mean"]["Train"], "o-", label="Train")
    ax.plot(data["MSE"]["mean"]['Test'], "o-", label="Test")
    
    # Add the error bands
    ax.fill_between(
        data["MSE"]["5%"]["Train"].index,
        data["MSE"]["5%"]["Train"],
        data["MSE"]["95%"]["Train"],
        alpha=0.25
    )
    
    ax.fill_between(
        data["MSE"]["5%"]["Test"].index,
        data["MSE"]["5%"]["Test"],
        data["MSE"]["95%"]["Test"],
        alpha=0.25
    )
    
    # Find the inflection point
    inflection = (data["MSE"]["mean"]["Test"] > data["MSE"]["mean"]['Test'].shift()).idxmax() - 1
    
    # Add to the figure
    ax.axvline(inflection, color="k", dashes=[2, 2])
    
    # Adjust some items
    sns.despine(offset=5)
    ax.set_ylabel("Mean Squared Error")
    ax.set_xlabel("Rank (i.e., Number of Factors)")
    ax.legend()
    plt.tight_layout()
    
    return fig

def rankcomp_figure(data, **kwargs):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    """
    fig, ax = plt.subplots(1)
    
    # Set some default args
    defaults = {
        "ax": ax,
        "errorbar": "sd",
        "errwidth": 0.75,
        "linewidth": 0.25
    }
    
    # Merge the two dicts together
    kwargs = kwargs | defaults
    
    # Plot away
    ax = sns.barplot(data=data, x="variable", y="value", hue='index', **kwargs)
    
    sns.despine(offset=1)
    
    ax.set(
        ylim=(0, 1), 
        yticks=np.linspace(0, 1, 11), 
        yticklabels=["0", "", "", "", "", "50", "", "", "", "", "100"],
        ylabel="Percent of Species Signal\nDescribed by Factor",
        xlabel=""
    )
    
    return fig