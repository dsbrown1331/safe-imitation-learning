# Efficient Probabilistic Performance Bounds for Inverse Reinforcement Learning
## Daniel S. Brown and Scott Niekum
### Follow the instructions below to reproduce results in our [AAAI 2018](https://arxiv.org/abs/1707.00724) and our [AAAI 2017 Fall Symposium](https://www.cs.utexas.edu/~dsbrown/pubs/Brown_AAAIFS17.pdf) papers.

 
  #### Dependencies
  - Matplotlib (for generating figures)
  - Python3 (for running scripts)
  
  #### Getting started
  - Make a build directory: `mkdir build`
  - Make a data directory to hold results: `mkdir data`
  
  #### Infinite Horizon GridWorld (Figure 2 in [AAAI 2018 paper](https://arxiv.org/abs/1707.00724))
  - Use `make gridworld_basic_exp` to build the experiment.
  - Execute `./gridworld_basic_exp` to run. Data will be output to `./data/gridworld`
  - Experiment will take some time to run since it runs 200 replicates for each number of demonstrations. Experiment parameters can be set in `src/gridWorldBasicExperiment.cpp`. 
  - Once experiment has finished run `python scripts/generateGridWorldBasicPlots.py` to generate figures used in paper.
  - You should get something similar to the following two plots

<div>
  <img src="figs/boundAccuracy.png" width="350">
  <img src="figs/boundError.png" width="350">
</div>
  
  
  #### Sensitivity to Confidence Parameter (Figure 3 in [AAAI 2018 paper](https://arxiv.org/abs/1707.00724))

  - Use `make gridworld_noisydemo_exp` to build the experiment.
  - Execute `./gridworld_noisydemo_exp` to run. Data will be output to `./data/gridworld_noisydemo_exp/`
  - Experiment will take some time to run since it runs 200 replicates for each number of demonstrations. Experiment parameters can be set in `src/gridWorldNoisyDemoExperiment.cpp`. 
  - Once experiment has finished run `python scripts/generateNoisyDemoPlots.py` to generate figures used in paper.
  - You should get something similar to the following two plots

<div>
  <img src="figs/noisydemo_accuracy_overAlpha.png" width="350">
  <img src="figs/noisydemo_bound_error_overAlpha.png" width="350">
</div>

   - Note that the bounds when c=0 are different than shown in paper. We are working on determining the reason for this discrepancy and it is probably due to an error in the original experiment execution and analysis.
  
  
  #### Comparison with theoretical bounds (Table 1 in in [AAAI 2018 paper](https://arxiv.org/abs/1707.00724))
  - UNDER CONSTRUCTION
  - Use `make gridworld_projection_exp` to build the experiment.
  - Execute `./gridworld_projection_exp` to run. Data will be output to `./data/abbeel_projection/`
  - Experiment will take some time to run since it runs 200 replicates for each number of demonstrations. Experiment parameters can be set in `src/gridWorldProjectionEvalExperiment.cpp`. 
  <!---  - Once experiment has finished run `python scripts/generateProjectionEvalTable.py` to generate table used in paper. -->
  
  
  #### Policy Selection for Driving Domain
  - UNDER CONSTRUCTION
  
  #### Policy Improvement
  - UNDER CONSTRUCTION
    
  
  

