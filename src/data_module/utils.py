import matplotlib.pyplot as plt
import pandas as pd
import os
ROMSA_TASKS = ["Pea_on_a_Peg", 'Post_and_Sleeve', 'Wire_Chaser']

def plot_ROMSA_progression(task:str, participant:str, save_path:str = "saved_figs/ROMSA_progression/", save:bool = True):
    assert task in ROMSA_TASKS, "task not in ROMSA tasks"
    print(int(participant[1:3]))
    assert int(participant[1:3]) in range(1, 13), "participant not in range"
    fig, ax = plt.subplots(ncols=3, figsize=(10, 10))
    path = '/scratch/wbg231/robotic_surgery_skill_assesment/data/processed_data/ROMSA/METADATA/scores.csv'
    df = pd.read_csv(path)
    ## only keep rows with continaing task and participant in file name col
    df = df[df['File_name'].str.contains(participant+"_"+task)]
    df.reset_index(inplace=True)
    print(df)
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
    
