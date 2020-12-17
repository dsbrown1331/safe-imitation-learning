#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/feature_brex.hpp"
#include <fstream>
#include <string>
#include <ctime>

//Compare proposed probabilistic upper bound on policy evaluation with worst-case bound

//Code to run experiment shown in Figure 2 in "Efficient Probabilistic Performance Bounds for Inverse Reinforcement Learning" from AAAI 2018.

using namespace std;

bool compareFunc(std::pair<int, float> &a, std::pair<int, float> &b)
{
    return a.second > b.second;
}

int main() 
{
    long seed = time(NULL);
    cout << "seed " << seed << endl;
    srand (seed);


    ////Experiment parameters
    const unsigned int numDemo = 20;            //number of demos to give
    const unsigned int rolloutLength = 15;          //max length of each demo
    const double alpha = 50; //50                    //confidence param for BIRL
    const unsigned int chain_length = 5000;//1000;//5000;        //length of MCMC chain
    const int sample_flag = 4;                      //param for mcmc walk type
    const int num_steps = 10;                       //tweaks per step in mcmc
    const bool mcmc_reject_flag = true;             //allow for rejects and keep old weights
    const double step = 0.01; //0.01
    const double min_r = -1;
    const double max_r = 1;
    bool stochastic = false;
    int burn = (int) 0.1 * chain_length;
    int skip = 5;
    double top_k = 0.25;  //use top 25% and bottom 25% as good and bad demos

    double eps = 0.001;
    
    //test arrays to get features
    const int numFeatures = 4; //white, red, blue, yellow, green
    const int numStates = 36;
    const int size = 6;
    double gamma = 0.95;
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);  
//    double** stateFeatures = random9x9GridNavGoalWorld();
//    double** stateFeatures = random9x9GridNavGoalWorld8Features();
    vector<unsigned int> initStates;
    for(int i=0;i<numStates;i++)
        initStates.push_back(i);
    vector<unsigned int> termStates = {};
    //termStates.push_back(numStates - 1);
    



    //create random world //TODO delete it when done


    double** stateFeatures = randomNxNGridNavWorldXFeatures(numStates, numFeatures);//debugDomain2x2(numStates, numFeatures);//random9x9GridNavGoalWorld8Features();//random9x9GridNavWorld8Features();
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);

    vector<vector<pair<unsigned int,unsigned int> > > trajectories; //used for feature counts
    vector<double> trajectory_gt_returns;
  
    ///  create a random weight vector with seed and increment of rep number so same across reps
    double* featureWeights = sample_unit_L1_norm(numFeatures);
    // featureWeights[0] = abs(featureWeights[0]);
    // for(int i=1; i<numFeatures; i++)
    //     featureWeights[i] = -abs(featureWeights[i]);
    //double featureWeights [3] = {-0.1, -0.7, -0.2};
        
    FeatureGridMDP fmdp(size, size, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    cout << "gamma " << fmdp.getDiscount() << endl;
    //delete[] featureWeights;
    ///  solve mdp for weights and get optimal policyLoss
    vector<unsigned int> opt_policy (fmdp.getNumStates());
    fmdp.valueIteration(eps);
    cout << "-- value function ==" << endl;
    fmdp.displayValues();
    // cout << "features" << endl;
    // displayStateColorFeatures(stateFeatures, size, size, numFeatures);
    cout << "Rewards" << endl;
    fmdp.displayRewards();
    fmdp.deterministicPolicyIteration(opt_policy);
    cout << "-- optimal policy --" << endl;
    fmdp.displayPolicy(opt_policy);
    fmdp.calculateQValues();
    cout << "-- feature weights --" << endl;
    fmdp.displayFeatureWeights();
    cout << endl;

    ///  generate numDemo demos from the initial state distribution
    trajectories.clear(); //used for feature counts
    trajectory_gt_returns.clear();
    for(unsigned int d = 0; d < numDemo; d++)
    {
       unsigned int s0 = initStates[d % initStates.size()];
       //cout << "demo from " << s0 << endl;
       //vector<pair<unsigned int, unsigned int>> traj = fmdp.monte_carlo_argmax_rollout(s0, rolloutLength);
       //Run random trajectories
       vector<pair<unsigned int, unsigned int>> traj = fmdp.epsilon_random_rollout(s0, rolloutLength, 1.0);
       
        cout << "trajectory " << d << endl;
        for(pair<unsigned int, unsigned int> p : traj)
            cout << "(" <<  p.first << "," << p.second << ")" << endl;
       double traj_return = fmdp.computeTrajectoryReturn(traj);
       cout << "return = " << traj_return << endl;
       trajectories.push_back(traj);
       trajectory_gt_returns.push_back(traj_return);

    }


    //compute pairwise preferences
    vector<pair<unsigned int, unsigned int> > pairwise_prefs;
    for(unsigned int i = 0; i < trajectories.size(); i++)
    {
        for(unsigned int j=i+1; j < trajectories.size(); j++)
        {
            if(trajectory_gt_returns[i] > trajectory_gt_returns[j])
            {
                pairwise_prefs.push_back(make_pair(j,i));
                cout << "traj " << i << " better than traj " << j << endl;
            }
            else if(trajectory_gt_returns[j] > trajectory_gt_returns[i])
            {
                pairwise_prefs.push_back(make_pair(i,j));
                cout << "traj " << j << " better than traj " << i << endl;
            }
            //TODO: try with and without this
            // else
            // {
            //     pairwise_prefs.push_back(make_pair(i,j));
            //     pairwise_prefs.push_back(make_pair(j,i));
            //     // cout << "trajs are equally good" << endl;
            // }
        }
    }
    cout << "================== SORTING ===================" << endl;
    //Try and sort the data to help BIRL
    std::vector<std::pair<int, float>> indexReturns;
    for(int i=0; i<numDemo; i++)
    {
        indexReturns.push_back(make_pair(i,trajectory_gt_returns[i]));
    }
    std::sort(indexReturns.begin(), indexReturns.end(), compareFunc);
    for(pair<int,float> p : indexReturns)
        cout << "index " << p.first << ", " << "return " << p.second << endl;
    
    //now it's sorted high to low so take the first 5 and the last 5 to run Yuchen's BIRL
    vector<vector<pair<unsigned int,unsigned int> > > good_trajectories; //used for yuchen birl
    vector<vector<pair<unsigned int,unsigned int> > > bad_trajectories; //used for yuchen birl
    int cnt = 0;
    for(pair<int,float> p : indexReturns)
    {
        int indx = p.first;
        if(cnt <  top_k * indexReturns.size())
        {
            good_trajectories.push_back(trajectories[indx]);
            cout << p.second << endl;
        }
        else if(cnt >= (1-top_k) * indexReturns.size())
        {
            bad_trajectories.push_back(trajectories[indx]);
            cout << p.second << endl;
        }
        cnt += 1;

    }
    cout << "good " << good_trajectories.size() << ", bad " << bad_trajectories.size() << endl;


    //Run B-REX

    



    clock_t begin = clock();
    FeatureBREX brex(&fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
    brex.addTrajectories(trajectories);
    brex.addPairwisePreferences(pairwise_prefs);
    brex.run();
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

cout << "----------" << endl;

    FeatureGridMDP* mapMDP = brex.getMAPmdp();
    cout << "weights learned by B-REX" << endl;
    mapMDP->displayFeatureWeights();
    cout << "True Feature Weights" << endl;
    for(int i=0; i<numFeatures; i++)
        cout << featureWeights[i] << ", ";
    cout << endl;
    cout << "Recovered reward" << endl;
    mapMDP->displayRewards();

    //solve for the optimal policy
    vector<unsigned int> eval_pi (mapMDP->getNumStates());
    mapMDP->valueIteration(eps);
    mapMDP->calculateQValues();
    mapMDP->getOptimalPolicy(eval_pi);
    cout << "-- optimal policy for MAP reward weights--" << endl;
    mapMDP->displayPolicy(eval_pi);
    double ploss = policyLoss(eval_pi, &fmdp);
    cout << "0-1 policy loss \%= " << ploss << endl;
    cout << "elapsed time = " << elapsed_secs << endl;

    cout << "----------" << endl;

    FeatureGridMDP* meanMDP = brex.getMeanMDP(burn, skip);
    cout << "weights learned by B-REX Mean" << endl;
    meanMDP->displayFeatureWeights();
    cout << "True Feature Weights" << endl;
    for(int i=0; i<numFeatures; i++)
        cout << featureWeights[i] << ", ";
    cout << endl;
    cout << "Recovered reward" << endl;
    meanMDP->displayRewards();

    //solve for the optimal policy
    vector<unsigned int> mean_pi (meanMDP->getNumStates());
    meanMDP->valueIteration(eps);
    meanMDP->calculateQValues();
    meanMDP->getOptimalPolicy(mean_pi);
    cout << "-- optimal policy for Mean reward weights--" << endl;
    meanMDP->displayPolicy(mean_pi);
    ploss = policyLoss(mean_pi, &fmdp);
    cout << "0-1 policy loss \%= " << ploss << endl;
    cout << "elapsed time = " << elapsed_secs << endl;

    cout << "-- true optimal policy --" << endl;
    fmdp.displayPolicy(opt_policy);
    cout << "True Rewards" << endl;
    fmdp.displayRewards();

    //TODO: Add BIRL code here too
    

    //delete world
    for(unsigned int s1 = 0; s1 < numStates; s1++)
    {
        delete[] stateFeatures[s1];
    }
    delete[] stateFeatures;


}


