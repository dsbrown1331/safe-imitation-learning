import numpy as np
import matplotlib.pyplot as plt
from numpy import nan, inf

#plot results for experiment7_1
#rewards are feasible in that all start states end up at goal within 25 steps

sample_flag = 4
chain_length = 5000
step = 0.01
alpha = 50
size = 9
num_reps = 100
rolloutLength = 20
numDemo = 20 #don't worry about one demos since ranking needs at least two!!
top_ks = [0.05, 0.1, 0.25, 0.5]
stochastic = 0



filePath = "data/brex_gridworld_ranked_yuchen/"



birl_ave_plosses = []
brex_ave_plosses = []

for top_k in top_ks:  
    birl_plosses = []
    brex_plosses = []
    if top_k == 0.25 or top_k == 0.05:
        padding = '0000'
    else:
        padding = '00000'
    filename = "GridWorldInfHorizon_numdemos" + str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) + "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) + "_rolloutLength" + str(rolloutLength) + "_stochastic" + str(stochastic) + "_topk" + str(top_k) + padding + ".txt"
    print(filename)
    f = open(filePath + filename,'r')   
    f.readline()                                #clear out comment from buffer
    for line in f:
        items = line.strip().split(",")
        birl_ploss = float(items[0])
        brex_ploss = float(items[1])

        birl_plosses.append(birl_ploss)
        brex_plosses.append(brex_ploss)
    birl_ave_plosses.append(np.mean(birl_plosses))
    brex_ave_plosses.append(np.mean(brex_plosses))

print("Yuchen Suboptimal Ranked Bayesian IRL vs. Suboptimal Ranked Bayesian REX")
for i,d in enumerate(top_ks):
    print("Demos = 20 top/bottom% = {}, BIRL Ploss = {:.3f}  BREX Ploss {:.3f}".format(d*100, birl_ave_plosses[i], brex_ave_plosses[i]))

percentage_improvements = []
for i,d in enumerate(top_ks):
    percentage_improvements.append((brex_ave_plosses[i] - birl_ave_plosses[i]) / birl_ave_plosses[i])

print("ave percentage improvement")
print(np.mean(percentage_improvements))


