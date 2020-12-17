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



filePath = "data/brex_gridworld_optnoise/"


birl_ave_plosses = []
brex_ave_plosses = []

for numDemo in numDemos:  
    birl_plosses = []
    brex_plosses = []
    print("=========", numDemo, "=========")
    filename = "GridWorldInfHorizon_numdemos" + str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) + "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) + "_rolloutLength" + str(rolloutLength) + "_stochastic" + str(stochastic) + ".txt"
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

print("Optimal Bayesian IRL vs. Optimal + Noise Bayesian REX")
for i,d in enumerate(numDemos):
    print("Demos = {}, BIRL Ploss = {:4f}  BREX Ploss {:.4f}".format(d, birl_ave_plosses[i], brex_ave_plosses[i]))

#print out as a table
print("Table")
print_str = ""
for i,d in enumerate(numDemos):
        print_str += "& {}".format(d)
print(print_str + "\\\\")



print_str = ""
print_str += "B-IRL "
for i,d in enumerate(numDemos):
        print_str += "& {:.3f}".format(birl_ave_plosses[i])
print(print_str + "\\\\")

print_str = ""
print_str += "B-REX "
for i,d in enumerate(numDemos):
        print_str += "& {:.3f}".format(brex_ave_plosses[i])
print(print_str + "\\\\")