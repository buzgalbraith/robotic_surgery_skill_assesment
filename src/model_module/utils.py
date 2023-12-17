import matplotlib.pyplot as plt
from model_module.self_supervised_model import Self_Supervised_Model
from model_module.base_model import Base_Model
import torch
import numpy as np

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
    plt.plot(target_loss, label="target loss")
    plt.plot(source_loss, label="source loss")
    plt.title(title)
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
    i = list(range(ci.shape[1]))
    fig, axes = plt.subplots (figsize=(20, 10))
    # y = torch.mean(ci, axis=0).flatten().tolist()
    # import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    
    ci = (torch.abs(mean-ci)).squeeze(2).tolist()
    mean = mean.flatten().tolist()
    # ci_a = (ci[1,:] - ci[0,:])/2
    # ci = ci_a.flatten().tolist()
    # ci_a = ci_a.flatten().tolist()
    # print(mean.flatten().tolist(), "mean")
    plt.scatter(i, mean, label='mean', color="black")
    plt.errorbar(i, mean, yerr=ci, fmt='none', color='red', capsize=5, label='Confidence Intervals')

    # fig, axes = plt.subplots(ncols=num_subplots, figsize=(20, 10))
    # ci_min = torch.min(ci[0,:])
    # ci_max = torch.max(ci[1,:])

    # ci_range = ci_max - ci_min
    # lower = ci_min - ci_range*0.1
    # upper = ci_max + ci_range*0.1
    # for i in range(num_subplots):
    #     axes[i].scatter(i, mean[i], label='mean', color="black")
    #     print(var[i], "variance")
    #     print("meain", mean[i])
    #     print("ci", ci[:,i])
    #     print("-"*100)
    #     ## set the y limits
    #     # print("lower", lower)
    #     # print("upper", upper)
    #     # print("diff", upper-lower)
    #     # axes[i].set_ylim([lower, upper])
    #     ## plot the fondidince interval
    #     axes[i].fill_between(i, ci[0, i], ci[1, i], alpha=0.4)
    #     axes[i].legend()
    if save_fig:
        save_path = save_path + title +'.png'
        print("fig saves", save_path)
        fig.savefig(save_path)
    else:
        plt.show()