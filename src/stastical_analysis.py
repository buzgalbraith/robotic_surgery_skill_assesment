from src.data_module.utils import get_partipant_metrics, get_partipant_input, get_paritpant_task_for_trail,get_trial_metrics, get_partipant_trials
from src.data_module.target_loader import Target_Loader
from src.model_module.self_supervised_model import Self_Supervised_Model
import numpy as np
import torch
from scipy import stats
INPUT_SIZE = 40
HIDDEN_SIZE = 16
NUM_STATES = 2
RESOLUTION = 1
STATES_TYPE = "last_and_first"
if __name__ == "__main__":
    self_supervised_model = Self_Supervised_Model(input_size=INPUT_SIZE, hidden_size =HIDDEN_SIZE,num_states=NUM_STATES, resolution=RESOLUTION, states_type=STATES_TYPE)
    self_supervised_model.load_model(mode="validate")   

    # task trial corelations 
    with open("results.txt", "w") as f:
        for i in range(1,13):
            if i < 10:
                X = "X0"+str(i)
            else:
                X = "X"+str(i)
            for task in ['Pea_on_a_Peg', 'Post_and_Sleeve', 'Wire_Chaser']:
                # print("Participant: ", X, " Task: ", task)
                y = get_partipant_metrics(task, X)   
                target_loader = Target_Loader(trials=[X], tasks=[task], batch_size=1)
                outs = []
                for x,_ in target_loader:
                    mean, var, ci = self_supervised_model.uncertainty_estimate(x, T=1000)
                    outs.append(mean.item())
                outs = np.array(outs)
                k = str.join(" ", task.split('_'))
                term_1 = "Participant " +  str(X[1:]) + " " + k  
                for y_type in ["Time", "Score", "Errors"]:
                    y_delta = y[y_type]
                    temp = np.corrcoef(y_delta, outs)[0,1]
                    if np.isnan(temp):
                        temp = 0
                    term_1 += "&" + str(round(temp, 3))
                term_1 += "\\\\ \hline"
                f.write(term_1)
                f.write("\n")
    # f.close()

    # ## variance task corelations 
    vals = []
    for i in range(1,7):
        X = "0"+str(i)
        file_names = get_paritpant_task_for_trail(X)
        metrics = get_trial_metrics(X)
        target_loader = Target_Loader(trials=file_names, batch_size=1)
        runnning_pred = 0
        for x, _ in target_loader:
            mean, var, ci = self_supervised_model.uncertainty_estimate(x, T=1000)
            runnning_pred += var.item()  
        vals.append(runnning_pred/len(file_names))
    vals = np.array(vals)
    imp = np.array(list(range(1,7)))
    temp = np.corrcoef(imp, vals)[0,1]
    ## mean participant corelations

        # ## variance task corelations 
    vals = []
    for i in range(1,7):
        X = "0"+str(i)
        file_names = get_paritpant_task_for_trail(X)
        metrics = get_trial_metrics(X)
        target_loader = Target_Loader(trials=file_names, batch_size=1)
        runnning_pred = 0
        for x, _ in target_loader:
            mean, var, ci = self_supervised_model.uncertainty_estimate(x, T=1000)
            runnning_pred += mean.item()  
        vals.append(runnning_pred/len(file_names))
    vals = np.array(vals)
    imp = np.array(list(range(1,7)))
    temp = np.corrcoef(imp, vals)[0,1]
    # student t test
    
    par_1 = get_partipant_trials("X01")
    par_3 = get_partipant_trials("X03")
    par_1_loader = Target_Loader(trials=par_1, batch_size=1)
    par_3_loader = Target_Loader(trials=par_3, batch_size=1)
    par_1_estimates = None
    for x, _ in par_1_loader:
        estimates = self_supervised_model.uncertainty_estimate(x, T=5000, get_estimates=True)
        if par_1_estimates is None:
            par_1_estimates = estimates
        else:
            par_1_estimates = torch.cat((par_1_estimates, estimates), dim=0)
    par_3_estimates = None
    for x, _ in par_3_loader:
        estimates = self_supervised_model.uncertainty_estimate(x, T=5000, get_estimates=True)
        if par_3_estimates is None:
            par_3_estimates = estimates
        else:
            par_3_estimates = torch.cat((par_3_estimates, estimates), dim=0)
    par_1_estimates = par_1_estimates.flatten().numpy()
    par_3_estimates = par_3_estimates.flatten().numpy()
    res = stats.ttest_ind(par_1_estimates, par_3_estimates, alternative='greater', equal_var=False)
    print(res)


    ## maybe do one way t test to see if person with genreally higher scores are given better higher values. 