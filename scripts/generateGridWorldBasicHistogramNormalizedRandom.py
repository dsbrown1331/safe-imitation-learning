import numpy as np
import matplotlib.pyplot as plt
import bound_methods
from numpy import nan, inf

#plot results for experiment7_1
#rewards are feasible in that all start states end up at goal within 25 steps

sample_flag = 4
chain_length = 10000
step = 0.01
alpha = 100
size = 9
num_reps = 10
rolloutLength = 100
numDemos = [1,3,5,7]
tol = 0.0001
gamma = 0.95
burn = 100
skip = 20
delta_conf = 0.95
stochastic = 1
#bounds = ["VaR 99"]#["VaR 99","VaR 95", "VaR 90"]
fmts = ['o-','s--','^-.', '*:','>-','d--']


filePath = "data/gridworld/"



for numDemo in numDemos:  
    true_perf_ratio = []
    predicted = []
    bound_error = []  
      
    print("=========", numDemo, "=========")
    for rep in range(num_reps):
        filename = "NormalizedRandomEVD_GridWorldInfHorizon_numdemos" + str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) + "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) + "_rolloutLength" + str(rolloutLength) + "_stochastic" + str(stochastic) + "_rep" + str(rep)+ ".txt"
        print(filename)
        f = open(filePath + filename,'r')   
        f.readline()                                #clear out comment from buffer
        actual = max(0.0, (float(f.readline()))) #get the true ratio, max with zero to avoid numerical issues that might result in negative EVD
        print("actual", actual)
        f.readline()                                #clear out ---
        wfcb = (float(f.readline())) #get the worst-case feature count bound
        f.readline()  #clear out ---
        samples = []
        for line in f:                              #read in the mcmc chain
            val = float(line)                       
            samples.append(float(line))
        #print samples
        #burn 
        burned_samples = samples[burn::skip]
        
        #print "max sample", np.max(burned_samples)
        #compute confidence bound
        print("min", min(burned_samples))
        print("max", max(burned_samples))
        plt.hist(burned_samples, 30, label="posterior")
        plt.hist([actual]*10, 100 , range=(min(max(actual - 0.1,0.0), max(min(burned_samples),0.0)),max(actual + 0.1, max(burned_samples))), label="actual")
        plt.legend()
        plt.title("num demos = {}  rep = {}".format(numDemo,rep))
        plt.show()


#    fig_cnt = 2
#    plt.figure(fig_cnt)
#    #plt.title(r"5x5 navigation domain, $\alpha = $" + str(alpha) + ", $step = $" + str(step))
#    plt.xlabel("number of demonstrations", fontsize=19)
#    plt.ylabel("accuracy", fontsize=19)
#    plt.plot(numDemos, accuracies, fmts[bounds.index(bound_type)], label= bound_type, lw = 2)
#    #plot 95% confidence line
#    #plt.plot([numDemos[0], numDemos[-1]],[0.95, 0.95], 'k--', lw=1)
#    plt.xticks(numDemos, fontsize=18) 
#    plt.yticks([0.84,0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0, 1.001],[0.84, 0.86,0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0,''], fontsize=18)
#    plt.legend(loc='lower right',fontsize=19)
#    plt.tight_layout()
#    plt.savefig("./figs/boundAccuracyNormalizedRandom.png") 
    



    

