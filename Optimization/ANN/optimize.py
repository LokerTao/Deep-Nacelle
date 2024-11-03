from HHO import HHO_optimizer
import torch
import numpy as np
import random

j = 12

torch.manual_seed(j)
torch.cuda.manual_seed(j)
torch.cuda.manual_seed_all(j)
random.seed(j)
np.random.seed(j)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

ANN_cd = torch.load("") # Load the best model from ../../Predict_Cd/ANN/

min_vals = [0.6, 1.1, 0.28, 9, 0.6, 1.1, 0.28, 9, 0.6, 1.1, 0.28, 9]
max_vals = [1.5, 1.3, 0.45, 15, 1.5, 1.3, 0.45, 15, 1.5, 1.3, 0.45, 15]

def func(x):

    ANN_cd.eval()

    with torch.no_grad():

        x = np.array(x).reshape(-1, 12)

        x = torch.from_numpy(x).float().to(device)

        y = ANN_cd(x)*0.205

    return y.detach().cpu().numpy()

best_position = []
best_cd = []

for i in range(20):

    s = HHO_optimizer(func, [0]*12, [1]*12, 12, 100, 500)

    s_ori = [s.bestIndividual[j]*(max_vals[j]-min_vals[j])+min_vals[j] for j in range(len(s.bestIndividual))]

    best_position.append(s_ori)
    best_cd.append(s.best)

np.savetxt("best_position", np.array(best_position))
np.savetxt("best_cd", np.array(best_cd).reshape(-1,1))