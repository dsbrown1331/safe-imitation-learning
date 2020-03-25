#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/feature_birl.hpp"
#include <fstream>
#include <string>

//Compare proposed probabilistic upper bound on policy evaluation
//Uses a normalized version of policy evaluation (V* - Veval) / (V* - Vpessimal)

//Code to run experiment shown in Figure 2 in "Efficient Probabilistic Performance Bounds for Inverse Reinforcement Learning" from AAAI 2018.

using namespace std;

int main() 
{

    ////Experiment parameters
    const unsigned int reps = 100;                    //repetitions per setting
    const vector<unsigned int> numDemos = {1,3,5,7};            //number of demos to give
    const vector<unsigned int> rolloutLengths = {100};          //max length of each demo
    const vector<double> alphas = {100}; //50                    //confidence param for BIRL
    const unsigned int chain_length = 10000;//1000;//5000;        //length of MCMC chain
    const int sample_flag = 4;                      //param for mcmc walk type
    const int num_steps = 10;                       //tweaks per step in mcmc
    const bool mcmc_reject_flag = true;             //allow for rejects and keep old weights
    const vector<double> steps = {0.01}; //0.01
    const double min_r = -1;
    const double max_r = 1;
    bool removeDuplicates = true;
    bool stochastic = true;

    int startSeed = 132;
    double eps = 0.001;
    
    //test arrays to get features
    const int numFeatures = 8; //white, red, blue, yellow, green
    const int numStates = 81;
    const int size = 9;
    double gamma = 0.9;
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);  
//    double** stateFeatures = random9x9GridNavGoalWorld();
//    double** stateFeatures = random9x9GridNavGoalWorld8Features();
    vector<unsigned int> initStates = {10, 13, 16, 37, 40, 43, 64, 67, 70};
    vector<unsigned int> termStates = {};
    
    
    //create directory for results
    string filePath = "./data/gridworld/";
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
            for(unsigned int rep = 0; rep < reps; rep++)
            {
                //set up file for output
                string filename = "NormalizedEVD_GridWorldInfHorizon_numdemos" +  to_string(numDemo) 
                                + "_alpha" + to_string((int)alpha) 
                                + "_chain" + to_string(chain_length) 
                                + "_step" + to_string(step)
                                + "_L1sampleflag" + to_string(sample_flag) 
                                + "_rolloutLength" + to_string(rolloutLength)
                                + "_stochastic" + to_string(stochastic)
                                + "_rep" + to_string(rep)+ ".txt";
                cout << filename << endl; 
                ofstream outfile(filePath + filename);
            
                srand(startSeed + 3*rep);
                cout << "------Rep: " << rep << "------" << endl;

                //create random world //TODO delete it when done
                double** stateFeatures = random9x9GridNavWorld8Features();
        
                vector<pair<unsigned int,unsigned int> > good_demos;
                vector<vector<pair<unsigned int,unsigned int> > > trajectories; //used for feature counts
              
                ///  create a random weight vector with seed and increment of rep number so same across reps
                double* featureWeights = sample_unit_L1_norm(numFeatures);
                double* negativeFeatureWeights = new double[numFeatures];
                for(int i=0; i < numFeatures; i++)
                    negativeFeatureWeights[i] = -featureWeights[i];
                    
                cout << "features" << endl;
                for(int i=0; i < numFeatures; i++)
                    cout << featureWeights[i] << ", ";
                cout << endl;
                
                cout << "negative features" << endl;
                for(int i=0; i < numFeatures; i++)
                    cout << negativeFeatureWeights[i] << ", ";
                cout << endl;
                
                   
                //normal mdp 
                FeatureGridMDP fmdp(size, size, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
                //minus mdp (reward is negative the true return so we can solve for the pessimal policy)
                FeatureGridMDP minus_fmdp(size, size, initStates, termStates, numFeatures, negativeFeatureWeights, stateFeatures, stochastic, gamma);
                delete[] featureWeights;
                delete[] negativeFeatureWeights;
                ///  solve mdp for weights and get optimal policyLoss
                vector<unsigned int> opt_policy (fmdp.getNumStates());
                fmdp.valueIteration(eps);
                //cout << "-- value function ==" << endl;
                //mdp.displayValues();
                cout << "features" << endl;
                displayStateColorFeatures(stateFeatures, size, size, numFeatures);
                //fmdp.deterministicPolicyIteration(opt_policy);
                fmdp.calculateQValues();
                fmdp.getOptimalPolicy(opt_policy);
                cout << "-- optimal policy --" << endl;
                fmdp.displayPolicy(opt_policy);
                cout << "-- feature weights --" << endl;
                fmdp.displayFeatureWeights();
                
                
                //solve for the pessimal policy via minus_fmdp
                vector<unsigned int> pessimal_pi (fmdp.getNumStates());
                minus_fmdp.valueIteration(eps);
                minus_fmdp.calculateQValues();
                minus_fmdp.getOptimalPolicy(pessimal_pi);
                cout << "-- pessimal policy --" << endl;
                minus_fmdp.displayPolicy(pessimal_pi);
                cout << "-- pessimal feature weights --" << endl;
                minus_fmdp.displayFeatureWeights();

                ///  generate numDemo demos from the initial state distribution
                trajectories.clear(); //used for feature counts
                for(unsigned int d = 0; d < numDemo; d++)
                {
                   unsigned int s0 = initStates[d];
                   //cout << "demo from " << s0 << endl;
                   vector<pair<unsigned int, unsigned int>> traj = fmdp.monte_carlo_argmax_rollout(s0, rolloutLength);
                   //cout << "trajectory " << d << endl;
                   //for(pair<unsigned int, unsigned int> p : traj)
                   //    cout << "(" <<  p.first << "," << p.second << ")" << endl;
                   trajectories.push_back(traj);
                }
                //put trajectories into one big vector for birl_test
                //weed out duplicate demonstrations
                good_demos.clear();
                for(vector<pair<unsigned int, unsigned int> > traj : trajectories)
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



                ///  run BIRL to get chain and Map policyLoss ///
                //give it a copy of mdp to initialize
                FeatureBIRL birl(&fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
                birl.addPositiveDemos(good_demos);
                birl.displayDemos();
                birl.run();
                FeatureGridMDP* mapMDP = birl.getMAPmdp();
                mapMDP->displayFeatureWeights();
                //cout << "Recovered reward" << endl;
                //mapMDP->displayRewards();

                //solve for the optimal policy
                vector<unsigned int> eval_pi (mapMDP->getNumStates());
                mapMDP->valueIteration(eps);
                mapMDP->calculateQValues();
                mapMDP->getOptimalPolicy(eval_pi);
//                cout << "-- value function ==" << endl;
//                mapMDP->displayValues();
//                mapMDP->deterministicPolicyIteration(map_policy);
//                cout << "-- optimal policy --" << endl;
                //mapMDP->displayPolicy(eval_pi);
                //cout << "\nPosterior Probability: " << birl.getMAPposterior() << endl;
                //double base_loss = policyLoss(eval_pi, &fmdp);
                //cout << "Current policy loss: " << base_loss << "%" << endl;

                /// We use the Map Policy as the evaluation policy

                
                
                //write actual, worst-case, and chain info to file

                ///compute actual expected return difference
                double Vstar = getExpectedReturn(&fmdp); //optimal policy expected value under true reward
                
                //compute the pesimal value under the true reward
                double Vmin = evaluateExpectedReturn(pessimal_pi, &fmdp, eps);
                //calculate (V^*_{R^*} - V^{\pi_eval}_{R^*}) / (V^*_{R^*} - V^{\pi^*(-R^*)}_{R^*})
                //this gives a normalized distance from optimal wrt to optimal-pessimal gap.
                double trueNormedDiff = (Vstar - evaluateExpectedReturn(eval_pi, &fmdp, eps)) / (Vstar - Vmin);
                cout << "True difference: " << trueNormedDiff << endl;
                outfile << "#true value --- wfcb --- mcmc ratios" << endl;
                outfile << trueNormedDiff << endl;
                outfile << "---" << endl;
                //compute worst-case feature count bound
                double wfcb = calculateWorstCaseFeatureCountBound(eval_pi, &fmdp, trajectories, eps);
                cout << "WFCB: " << wfcb << endl;
                outfile << wfcb << endl;
                outfile << "---" << endl;

                
                 //Calculate differences and output them to file in format true\n---\ndata
                for(unsigned int i=0; i<chain_length; i++)
                {
                    //cout.precision(5);
                    //get sampleMDP from chain
                    FeatureGridMDP* sampleMDP = (*(birl.getRewardChain() + i));
                    //((FeatureGridMDP*)sampleMDP)->displayFeatureWeights();
                    //cout << "===================" << endl;
                    //cout << "Reward " << i << endl;
                    //sampleMDP->displayRewards();
                    //cout << "--------" << endl;
                    vector<unsigned int> sample_pi(sampleMDP->getNumStates());
                    //cout << "sample opt policy" << endl;
                    sampleMDP->getOptimalPolicy(sample_pi);
                    //sampleMDP->displayPolicy(sample_pi);
                    //cout << "Value" << endl;
                    //sampleMDP->displayValues();
                    double Vstar = getExpectedReturn(sampleMDP);
                    //cout << "True Exp Val" << endl;
                    //cout << Vstar << endl;
                    //cout << "Eval Policy" << endl; 
                    double Vhat = evaluateExpectedReturn(eval_pi, sampleMDP, eps);
                    //cout << Vhat << endl;
                    
                    //calculate pessimal policy
                    vector<unsigned int> pessimal_sample_pi(sampleMDP->getNumStates());
                    //need to create new mdp with -R 
                    double* sample_weights = sampleMDP->getFeatureWeights();
                    double* pessimal_weights = new double[numFeatures];
                    for(int i=0; i < numFeatures; i++)
                        pessimal_weights[i] = -sample_weights[i];
                    
                    FeatureGridMDP minus_mdp(sampleMDP->getGridWidth(), sampleMDP->getGridHeight(), sampleMDP->getInitialStates(), sampleMDP->getTerminalStates(), sampleMDP->getNumFeatures(), pessimal_weights , sampleMDP->getStateFeatures(), sampleMDP->isStochastic(), sampleMDP->getDiscount());
                    //set the walls, if any
                    minus_mdp.setWallStates(sampleMDP->getWallStates());
                    //solve for pessimal policy
                    minus_mdp.valueIteration(eps);
                    minus_mdp.getOptimalPolicy(pessimal_sample_pi);
                    double Vmin = evaluateExpectedReturn(pessimal_sample_pi, sampleMDP, eps);
                    delete[] pessimal_weights;
                    
                    double VNormedDiff = (Vstar - Vhat) / (Vstar - Vmin);
                    //cout << "pred diff: " << VNormedDiff << endl;
                    outfile << VNormedDiff << endl;
                }    
           

                //delete world
                for(unsigned int s1 = 0; s1 < numStates; s1++)
                {
                    delete[] stateFeatures[s1];
                }
                delete[] stateFeatures;

            }

        }

        }
    }
}


}


