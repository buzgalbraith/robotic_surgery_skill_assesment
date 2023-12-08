import matplotlib.pyplot as plt
from model_module.self_supervised_model import Self_Supervised_Model
from model_module.base_model import Base_Model
import torch
import seaborn as sb

def plot_loss_history(loss_history:list, title:str, save_fig:bool = True, save_path:str = "saved_figs/")->None:
    """plots loss over time.
    Args:
        loss_history(list): list of loss values
        title(str): title of plot
        save_fig(bool, optional): if true, saves figure to save_path
        save_path(str, optional): path to save figure
    Returns:
        None
    """
    # import ipdb; ipdb.set_trace()
    # sb.lineplt
    # plt.scatter(loss_history)
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if save_fig:
        save_path = save_path + title + '.png'
        plt.savefig(save_path )
    else:
        plt.show()
def plot_target_and_source_loss(target_loss:list, source_loss:list, title:str, save_fig:bool = True, save_path:str = "saved_figs/")->None:
    """plots loss over time.
    Args:
        target_loss(list): list of target loss values
        source_loss(list): list of source loss values
        title(str): title of plot
        save_fig(bool, optional): if true, saves figure to save_path
        save_path(str, optional): path to save figure
    Returns:
        None
    """
    # import ipdb; ipdb.set_trace()
    # sb.lineplt
    # plt.scatter(loss_history)
    # plt.plot(loss_history)
    plt.plot(target_loss, label="target loss")
    plt.plot(source_loss, label="source loss")
    plt.title(title)
    # plt.ylim((0,100))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if save_fig:
        save_path = save_path + title + '.png'
        plt.savefig(save_path )
    else:
        plt.show()
def plot_confidence_interval_supervised(model:Self_Supervised_Model, x:torch.tensor, y:torch.tensor, save_fig:bool=True, save_path="saved_figs/", title="confidence_interval"):
    mean, var, ci = model.uncertainty_estimate(x)
    num_subplots = ci.shape[1]
    fig, axes = plt.subplots(ncols=num_subplots, figsize=(20, 10))
    for i in range(num_subplots):
        col = 'blue' if y[i] == 1 else 'red'
        axes[i].scatter(i, mean[i], label='mean', color="black")
        axes[i].fill_between(i, ci[0, i], ci[1, i], alpha=0.95, label='95% confidence interval', color=col)
        axes[i].set_title("blue if expert, red if novice")
        axes[i].set_ylim([0, 1])
        axes[i].legend()
    if save_fig:
        save_path = save_path + title + '.png'
        fig.savefig(save_path)
    else:
        plt.show()
def plot_confidence_interval_unsupervised(model:Self_Supervised_Model, x:torch.tensor,labels:list, T=1000,save_fig:bool=True, save_path="saved_figs/", title="confidence_interval"):
    
    mean, var, ci = model.uncertainty_estimate(x, T=T)
    num_subplots = ci.shape[1]
    fig, axes = plt.subplots(ncols=num_subplots, figsize=(20, 10))
    if mean.shape[0]>1:
        for i in range(num_subplots):
            axes[i].scatter(i, mean[i], label='mean', color="black")
            axes[i].fill_between(i, ci[0, i], ci[1, i], alpha=0.95, label=labels[i])
            # axes[i].set_title(labels[i])
            axes[i].set_ylim([0, 1])
            axes[i].legend()
    else:
        i = 0
        axes.scatter(i, mean[i], label='mean', color="black")
        axes.fill_between(i, ci[0, i], ci[1, i], alpha=0.95, label=labels[i])
        # axes.set_title(labels[i])
        axes.set_ylim([0, 1])
        axes.legend()

    if save_fig:
        save_path = save_path + title +'.png'
        fig.savefig(save_path)
    else:
        plt.show()