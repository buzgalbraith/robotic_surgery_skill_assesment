"""This file is specifically for converting the JIGSAW and ROMSA data to csvs and downsampleing them  with the correct colnames.
"""
import os
import pandas as pd

DATA_PATH = 'data/original_data/'
SAVE_PATH = 'data/processed_data/'
JIGSAW_PATH = os.path.join(DATA_PATH, 'JIGSAW/')
JIGSAW_TASKS = ['Suturing', 'Knot_Tying', 'Needle_Passing']
ROMSA_Path = os.path.join(DATA_PATH, 'ROMSA/')
ROMSA_TASKS = ["Pea_on_a_Peg", 'Post_and_Sleeve', 'Wire_Chaser']


def downsample(df:pd.DataFrame, original_frequency:int) -> pd.DataFrame:
    """here we are downsampling the data to 1Hz.  (could play with this rate later, so far just using 1hz since that is what they used in the paper)
        it is also worth noting that we are just taking every nth instead of specifically checking that we get the first row in each second. 
    Args: 
        df (pd.DataFrame): df to downsample
        original_frequency (int): frequency of the original df (in Hz). (JIGSAW is 30Hz, ROMSA is 50hz)
    Returns:
        df (pd.DataFrame): downsampled df
    """
    df = df.iloc[::original_frequency, :] ## take only every nth col 
    return df

def format_ROMSA(read_path:str, save_path:str)->None:
    """reads a ROMSA file and saves a csv with the correct colnames.
    Args:
        read_path (str): path to the ROMSAI file
        save_path (str): path to save the df
    Returns:
        None 
    """
    print("reading file: ", read_path)
    df = pd.read_csv(read_path)
    ## remove unnecessary cols
    to_drop = [col for col in df.columns if 'joint' in col]
    to_drop += [col for col in df.columns if 'wrench' in col]
    to_drop += [col for col in df.columns if 'orientation_x' in col]
    to_drop += [col for col in df.columns if 'orientation_y' in col]
    to_drop += [col for col in df.columns if 'orientation_z' in col]
    df.drop(to_drop, axis=1, inplace=True)
    ## order cols
    colnames = []
    for part in ["MTML" , "MTMR", "PSM1", "PSM2"]:
        for col in ["position"]:
            for dim in ["x", "y", "z"]:
                colnames.append(part + "_" + col+ "_" + dim)
        for col in ["linear", "angular"]:
            for dim in ["x", "y", "z"]:
                colnames.append(part + "_velocity_" + col + "_" + dim)
        colnames.append(part + "_orientation_w")
    df = df[colnames]
    df = downsample(df, 50) ## downsample to 1Hz
    df.to_csv(save_path, index=False)
    print("saved to: ", save_path)
    print("done")
def get_ROMSA_csvs(romsa_path:str, save_path:str, tasks:list=ROMSA_TASKS)->None:
    """converts ROMSA files to csvs with the correct colnames.
    Args:
        romsa_path (str): path to the ROMSA directory
        save_path (str): path to save the csvs
        tasks (list): list of ROMSA tasks to convert
    Returns:
        None
    """
    try:
        os.mkdir(save_path + "ROMSA/")
    except:
        pass
    for task in tasks:
        task_path = romsa_path + task 
        task_save = save_path + "ROMSA/" +task + '/'
        try:
            os.mkdir(task_save)
        except:
            pass
        for file in os.listdir(task_path):
            if file.endswith('.csv'):
                read_path = os.path.join(task_path, file)
                print("task save: ", task_save)
                file_save = task_save + file[:-4] + '.csv'
                format_ROMSA(read_path, file_save)


def get_col_names()->list:
    """hardcoded colnames for JIGSAW data.
    Args:
        None
    Returns: 
        colnames (list) : generated colnames 
    """
    colnames = []
    for part in ["MTML", "MTMR", "PSM1", "PSM2"]:
        for col in ["position"]:
            for dim in ["x", "y", "z"]:
                colnames.append(part + "_" + col+ "_" + dim)
        for col in ["rotation"]:
            for element in range(0,9):
                colnames.append(part + "_" + col + "_" + str(element))
        for col in ["linear", "angular"]:
            for dim in ["x", "y", "z"]:
                colnames.append(part + "_velocity_" + col + "_" + dim)
        for col in ["orientation_w"]:
            colnames.append(part + "_" + col)
    return colnames

def format_JIGSAW(read_path:str, save_path:str)->None:
    """reads a JIGSAW file and returns a df with the correct colnames.
    Args:
        read_path (str): path to the JIGSAW file
        save_path (str): path to save the df
    Returns:
        None 
    """
    print("reading file: ", read_path)
    with open(read_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split('    ') for line in lines]
        df = pd.DataFrame(lines)
        df = df.astype(float)
        df.columns = get_col_names()
    f.close()
    to_drop = [col for col in df.columns if 'rotation' in col]
    df.drop(to_drop, axis=1, inplace=True)
    df = downsample(df, 30) ## downsample to 1Hz
    df.to_csv(save_path, index=False)
    print("saved to: ", save_path)
    print("done")
def get_JIGSAW_csvs(jigsaw_path:str, save_path:str, tasks:list=JIGSAW_TASKS)->None:
    """converts JIGSAW files to csvs with the correct colnames.
    Args:
        jigsaw_path (str): path to the JIGSAW directory
        save_path (str): path to save the csvs
        tasks (list): list of JIGSAW tasks to convert
    Returns:
        None
    """
    try:
        os.mkdir(save_path + "JIGSAW/")
    except:
        pass
    for task in tasks:
        task_path = jigsaw_path + task +'/kinematics/AllGestures/'
        task_save = save_path + "JIGSAW/" + task + '/'
        try:
            os.mkdir(task_save)
        except:
            pass
        for file in os.listdir(task_path):
            if file.endswith('.txt'):
                read_path = os.path.join(task_path, file)
                print("task save: ", task_save)
                file_save = task_save + file[:-4] + '.csv'
                format_JIGSAW(read_path, file_save)

def get_JIGSAW_ground_truth(read_path, save_path, task_list):
    """reads a JIGSAW ground truth file and returns a df with the correct colnames.
    Args:
        read_path (str): path to the JIGSAW ground truth file
        save_path (str): path to save the df
        task_list (list): list of JIGSAW tasks to convert
    Returns:
        None 
    """
    save_path = save_path + "JIGSAW/METADATA/"
    try:
        os.mkdir(save_path)
    except: 
        pass
    for task in task_list:
        task_read = read_path + "JIGSAW/" + task + "/meta_file_" + task + '.txt'
        task_save = save_path + task + '.csv'
        print("reading file: ", task_read)
        with open(task_read, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split('\t') for line in lines]
            lines = [[word for word in line if word != ""] for line in lines]
            lines = lines[:-1]
            df = pd.DataFrame(lines)
            df.columns = ['task', 'robotic_surgery_experience', 'overall_score', 'score_component_1','score_component_2','score_component_3','score_component_4','score_component_5','score_component_6']
        f.close()
        df.to_csv(task_save, index=False)
        print("saved to: ", task_save)
        print("done")

def get_ROMSA_ground_truth(read_path, save_path):
    """reads a ROMSA ground truth file and returns a df with the correct colnames.
        We do not need to edit this from how you download it, so this more or less just copies the file to the correct location.
    Args:
        read_path (str): path to the ROMSA ground truth file
        save_path (str): path to save the df
    Returns:
        None 
    """
    save_path = save_path + "ROMSA/METADATA/"
    read_path  = read_path + "ROMSA/scores.csv"
    try:
        os.mkdir(save_path)
    except: 
        pass
    df = pd.read_csv(read_path)
    df.drop(['Unnamed: 4'], axis=1, inplace=True)
    df.to_csv(save_path + 'scores.csv', index=False)

if __name__ == "__main__":
    # make directory for generated data
    os.makedirs(SAVE_PATH, exist_ok=True)
    get_JIGSAW_csvs(JIGSAW_PATH, SAVE_PATH,JIGSAW_TASKS)
    get_ROMSA_csvs(ROMSA_Path, SAVE_PATH, ROMSA_TASKS)
    get_JIGSAW_ground_truth(DATA_PATH, SAVE_PATH, JIGSAW_TASKS)
    get_ROMSA_ground_truth(DATA_PATH, SAVE_PATH)
    