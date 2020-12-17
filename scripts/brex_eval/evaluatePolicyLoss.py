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
numDemos = [2,5,10,20,30] #don't worry about one demos since ranking needs at least two!!
stochastic = 0



filePath = "data/brex_gridworld_optsubopt/"


ave_plosses = []

for numDemo in numDemos:  
    ploss = []
    print("=========", numDemo, "=========")
    filename = "GridWorldInfHorizon_numdemos" + str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) + "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) + "_rolloutLength" + str(rolloutLength) + "_stochastic" + str(stochastic) + ".txt"
    print(filename)
    f = open(filePath + filename,'r')   
    f.readline()                                #clear out comment from buffer
    for line in f:
        items = line.strip().split(",")
        birl_ploss = float(items[0])
        brex_ploss = float(items[1])
        ratio_increase_over_birl = (brex_ploss - birl_ploss) / birl_ploss
        ploss.append(ratio_increase_over_birl)
    ave_plosses.append(np.mean(ploss))

print("Optimal Bayesian IRL vs. Suboptimal Bayesian REX")
for i,d in enumerate(numDemos):
    print("Demos = {}, Performance is {}% worse than Bayesian IRL".format(d, ave_plosses[i]*100))
