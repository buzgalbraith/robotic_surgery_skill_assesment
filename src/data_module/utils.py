import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
ROMSA_TASKS = ["Pea_on_a_Peg", 'Post_and_Sleeve', 'Wire_Chaser']
scores_path = "data/processed_data/ROMSA/METADATA/scores.csv"

def plot_ROMSA_progression(task:str, participant:str, save_path:str = "saved_figs/ROMSA_progression/", save:bool = True):
    """Plots the ground truth of a single participant on a single task over repeated trials.
    args:
        task (str): task to consider 
        participant (str) : participant id 
        save_path (str, optional): path to save figure
        save (bool, optinla): whether to save figure or not

    """
    assert task in ROMSA_TASKS, "task not in ROMSA tasks"
    assert int(participant[1:3]) in range(1, 13), "participant not in range"
    fig, ax = plt.subplots(ncols=3, figsize=(10, 10))
    path = './data/processed_data/ROMSA/METADATA/scores.csv'
    df = pd.read_csv(path)
    ## only keep rows with continaing task and participant in file name col
    df = df[df['File_name'].str.contains(participant+"_"+task)]
    df.reset_index(inplace=True)
    ax[0].plot(df['Score'], label = "ROMSA score")
    ax[1].plot(df['Errors'], label = "Errors")
    ax[2].plot(df['Time'], label = "Time")
    ax[0].set_xlabel("Trial")
    ax[1].set_xlabel("Trial")
    ax[2].set_xlabel("Trial")
    ax[0].set_ylabel("Score")
    ax[1].set_ylabel("Errors")
    ax[2].set_ylabel("Time to Completion")
    plt.tight_layout()
    if save:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path+task+"_"+participant+".png")
    else:
        plt.show()
def plot_ROMSA_progression_and_model_estimate(model, input, task:str, participant:str, T=10000,save_path:str = "saved_figs/validate/", save:bool = True):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 10))
    path = './data/processed_data/ROMSA/METADATA/scores.csv'
    df = pd.read_csv(path)
    ## only keep rows with continaing task and participant in file name col
    df = df[df['File_name'].str.contains(participant+"_"+task)]
    df.reset_index(inplace=True)
    ax[0].plot(df['Score'], label = "ROMSA score")
    ax[0].set_ylabel("Score")
    ax[0].set_xlabel("Trial")
    mean, var, ci = model.uncertainty_estimate(input, T=T)
    i = list(range(ci.shape[1]))
    
    ci = (torch.abs(mean-ci)).squeeze(2).tolist()
    mean = mean.flatten().tolist()
    ax[1].scatter(i, mean, label='mean', color="black")
    ax[1].errorbar(i, mean, yerr=ci, fmt='none', color='red', capsize=5, label='Confidence Intervals')
    ax[1].set_ylabel("Prediction of expert class")
    ax[1].set_xlabel("Trial")
    fig.suptitle(task+"_"+participant)
    plt.tight_layout()
    if save:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path+task+"_"+participant+".png")
    else:
        plt.show()


def get_paritpant_task_for_trail(trial_name):
    trial_name = "_"+trial_name
    df = pd.read_csv(scores_path)
    df = df.loc[df['File_name'].str.contains(trial_name)]
    return df["File_name"].tolist()
def get_trial_metrics(trials):
    trial_name = "_"+trials
    df = pd.read_csv(scores_path)
    df = df.loc[df['File_name'].str.contains(trial_name)]
    return df
def get_partipant_trials(participant_id):
    df = pd.read_csv(scores_path)
    df = df.loc[df['File_name'].str.contains(participant_id)]
    return df["File_name"].tolist()
def get_task_partipats(task_name):
    df = pd.read_csv(scores_path)
    df = df.loc[df['File_name'].str.contains(task_name)]
    return df["File_name"].tolist()
def get_trials_task_partipants(task_name, participant_id):
    df = pd.read_csv(scores_path)
    df = df.loc[df['File_name'].str.contains(task_name)]
    df = df.loc[df['File_name'].str.contains(participant_id)]
    return df["File_name"].tolist()
def get_partipant_metrics(task_name, participant_id):
    df = pd.read_csv(scores_path)
    df = df.loc[df['File_name'].str.contains(task_name)]
    df = df.loc[df['File_name'].str.contains(participant_id)]
    return df
def get_partipant_input(task_name, participant_id):
   df = get_trials_task_partipants(task_name, participant_id)
   path = "data/processed_data/ROMSA/"+task_name+"/"
   data = []
   for trial in df:
       df = pd.read_csv(path+trial+".csv")
       data.append(torch.Tensor(df.values).reshape(1, -1, 40))
   return data

