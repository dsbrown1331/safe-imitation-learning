#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/feature_birl.hpp"
#include "../include/feature_brex.hpp"
#include <fstream>
#include <string>

//Compare proposed probabilistic upper bound on policy evaluation with worst-case bound

//Code to run experiment shown in Figure 2 in "Efficient Probabilistic Performance Bounds for Inverse Reinforcement Learning" from AAAI 2018.

using namespace std;

bool compareFunc(std::pair<int, float> &a, std::pair<int, float> &b)
{
    return a.second > b.second;
}

int main() 
{

    ////Experiment parameters
    const unsigned int reps = 100;                    //repetitions per setting
    const vector<unsigned int> numDemos = {20};            //number of demos to give
    const vector<unsigned int> rolloutLengths = {20};          //max length of each demo
    const vector<double> alphas = {50}; //50                    //confidence param for BIRL
    const unsigned int chain_length = 5000;//1000;//5000;        //length of MCMC chain
    const int sample_flag = 4;                      //param for mcmc walk type
    const int num_steps = 10;                       //tweaks per step in mcmc
    const bool mcmc_reject_flag = true;             //allow for rejects and keep old weights
    const vector<double> steps = {0.01}; //0.01
    const double min_r = -1;
    const double max_r = 1;
    bool removeDuplicates = true;
    bool stochastic = false;
    int burn = (int) 0.1 * chain_length;
    int skip = 5;
    double top_k = 0.05;

    int startSeed = 132;
    double eps = 0.001;
    
    //test arrays to get features
    const int numFeatures = 4; //white, red, blue, yellow, green
    const int numStates = 36;
    const int size = 6;
    double gamma = 0.9; 
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);  
//    double** stateFeatures = random9x9GridNavGoalWorld();
//    double** stateFeatures = random9x9GridNavGoalWorld8Features();
    vector<unsigned int> initStates;
    for(int i=0;i<numStates;i++)
        initStates.push_back(i);
    
    vector<unsigned int> termStates = {};
    
    
    //create directory for results
    string filePath = "./data/brex_gridworld_ranked_yuchen/";
    string mkdirFilePath = "mkdir -p " + filePath;
    system(mkdirFilePath.c_str());

for(unsigned int rolloutLength : rolloutLengths)
{
    //iterate over alphas
    for(double alpha : alphas)
    {
        cout << "======Alpha: " << alpha << "=====" << endl;
        //iterate over number of demonstrations
        for(unsigned int numDemo : numDemos)
        {
            cout << "****Num Demos: " << numDemo << "****" << endl;
            //iterate over repetitions
        for(double step : steps)
        {
            //set up file for output
                string filename = "GridWorldInfHorizon_numdemos" +  to_string(numDemo) 
                                + "_alpha" + to_string((int)alpha) 
                                + "_chain" + to_string(chain_length) 
                                + "_step" + to_string(step)
                                + "_L1sampleflag" + to_string(sample_flag) 
                                + "_rolloutLength" + to_string(rolloutLength)
                                + "_stochastic" + to_string(stochastic)
                                + "_topk" + to_string(top_k)
                                + ".txt";
                cout << filename << endl; 
                ofstream outfile(filePath + filename);
            
            for(unsigned int rep = 0; rep < reps; rep++)
            {
                
                srand(startSeed + 3*rep);
                cout << "------Rep: " << rep << "------" << endl;

                //create random world //TODO delete it when done
                double** stateFeatures = randomNxNGridNavWorldXFeatures(numStates, numFeatures);
        
                vector<pair<unsigned int,unsigned int> > good_demos;
                vector<pair<unsigned int,unsigned int> > bad_demos;
                vector<vector<pair<unsigned int,unsigned int> > > trajectories; //used for feature counts
                vector<double> trajectory_gt_returns;
              
                ///  create a random weight vector with seed and increment of rep number so same across reps
                double* featureWeights = sample_unit_L1_norm(numFeatures);
                    
                FeatureGridMDP fmdp(size, size, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
                delete[] featureWeights;
                ///  solve mdp for weights and get optimal policyLoss
                vector<unsigned int> opt_policy (fmdp.getNumStates());
                fmdp.valueIteration(eps);
                //cout << "-- value function ==" << endl;
                //mdp.displayValues();
                //cout << "features" << endl;
                //displayStateColorFeatures(stateFeatures, size, size, numFeatures);
                fmdp.deterministicPolicyIteration(opt_policy);
                cout << "-- optimal policy --" << endl;
                fmdp.displayPolicy(opt_policy);
                fmdp.calculateQValues();
                cout << "-- feature weights --" << endl;
                fmdp.displayFeatureWeights();
                cout << "Rewards" << endl;
                fmdp.displayRewards();

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
                
                    // cout << "trajectory " << d << endl;
                    // for(pair<unsigned int, unsigned int> p : traj)
                    //     cout << "(" <<  p.first << "," << p.second << ")" << endl;
                double traj_return = fmdp.computeTrajectoryReturn(traj);
                //cout << "return = " << traj_return << endl;
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
                            //cout << "traj " << i << " better than traj " << j << endl;
                        }
                        else if(trajectory_gt_returns[j] > trajectory_gt_returns[i])
                        {
                            pairwise_prefs.push_back(make_pair(i,j));
                            //cout << "traj " << j << " better than traj " << i << endl;
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


                //put trajectories into one big vector for birl_test
                //weed out duplicate demonstrations
                good_demos.clear();
                for(vector<pair<unsigned int, unsigned int> > traj : good_trajectories)
                    for(pair<unsigned int, unsigned int> p : traj)
                        if(removeDuplicates)
                        {
                            if(std::find(good_demos.begin(), good_demos.end(), p) == good_demos.end())
                                good_demos.push_back(p);
                        }
                        else
                        {    
                            good_demos.push_back(p);
                        }

                bad_demos.clear();
                for(vector<pair<unsigned int, unsigned int> > traj : bad_trajectories)
                    for(pair<unsigned int, unsigned int> p : traj)
                        if(removeDuplicates)
                        {
                            if(std::find(good_demos.begin(), bad_demos.end(), p) == bad_demos.end())
                                bad_demos.push_back(p);
                        }
                        else
                        {    
                            bad_demos.push_back(p);
                        }


                ///  run BIRL to get chain and Map policyLoss ///
                //give it a copy of mdp to initialize
                FeatureBIRL birl(&fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
                birl.addPositiveDemos(good_demos);
                birl.addNegativeDemos(bad_demos);
                birl.displayDemos();
                birl.run();
                cout << "birl complete" << endl;
                FeatureGridMDP* meanMDP = birl.getMeanMDP(burn, skip);
                meanMDP->displayFeatureWeights();
                //cout << "Recovered reward" << endl;
                //mapMDP->displayRewards();

                //solve for the optimal policy
                vector<unsigned int> birl_eval_pi (meanMDP->getNumStates());
                meanMDP->valueIteration(eps);
                meanMDP->calculateQValues();
                meanMDP->getOptimalPolicy(birl_eval_pi);

                ///compute actual expected return difference
                double birltrueDiff = abs(getExpectedReturn(&fmdp) - evaluateExpectedReturn(birl_eval_pi, &fmdp, eps));
                /// We use the Map Policy as the evaluation policy

                
                ///Run Bayesian REX to compare

                FeatureBREX brex(&fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
                brex.addTrajectories(trajectories);
                brex.addPairwisePreferences(pairwise_prefs);
                brex.run();

                FeatureGridMDP* brexmeanMDP = brex.getMeanMDP(burn, skip);
                cout << "weights learned by B-REX Mean" << endl;
                brexmeanMDP->displayFeatureWeights();
                cout << "True Feature Weights" << endl;
                for(int i=0; i<numFeatures; i++)
                    cout << featureWeights[i] << ", ";
                cout << endl;
                cout << "Recovered reward" << endl;
                brexmeanMDP->displayRewards();

                //solve for the optimal policy
                vector<unsigned int> mean_pi (brexmeanMDP->getNumStates());
                brexmeanMDP->valueIteration(eps);
                brexmeanMDP->calculateQValues();
                brexmeanMDP->getOptimalPolicy(mean_pi);
                cout << "-- optimal policy for Mean reward weights--" << endl;
                brexmeanMDP->displayPolicy(mean_pi);

                cout << "-- true optimal policy --" << endl;
                fmdp.displayPolicy(opt_policy);
                cout << "True Rewards" << endl;
                fmdp.displayRewards();

                double brextrueDiff = abs(getExpectedReturn(&fmdp) - evaluateExpectedReturn(mean_pi, &fmdp, eps));
                
                cout << "True difference: birl=" << birltrueDiff << ", brex=" << brextrueDiff << endl;
                if(rep == 0)
                    outfile << "#true policy loss value birl, true policy loss value diff brex" << endl;
                outfile << birltrueDiff << ", " << brextrueDiff << endl;
                

                //delete world
                for(unsigned int s1 = 0; s1 < numStates; s1++)
                {
                    delete[] stateFeatures[s1];
                }
                delete[] stateFeatures;

            }
            outfile.close();
        }

        }
        
    }
}


}


