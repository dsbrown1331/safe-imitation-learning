
#ifndef feature_brex_h
#define feature_brex_h

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <math.h>
#include "mdp.hpp"
#include "../include/unit_norm_sampling.hpp"

using namespace std;

class FeatureBREX { // B-REX with known features
      
   protected:
      
      double r_min, r_max, step_size;
      unsigned int chain_length;
      unsigned int grid_height, grid_width;
      double alpha;
      unsigned int iteration;
      int sampling_flag;
      bool mcmc_reject; //If false then it uses Yuchen's sample until accept method, if true uses normal MCMC sampling procedure
      int num_steps; //how many times to change current to get proposal
      unsigned int nfeatures;
      double gamma;

      double** stateFeatures;
      vector< vector<double> > fcounts;  //TODO add fcounts to this everytime a demo is added
      vector<pair<unsigned int, unsigned int> > pairwise_preferences; //vector or pairwise preferences

      
      void initializeMDP();
      void modifyFeatureWeightsRandomly(FeatureGridMDP * gmdp, double step_size);
      void sampleL1UnitBallRandomly(FeatureGridMDP * gmdp);
      void updownL1UnitBallWalk(FeatureGridMDP * gmdp, double step);
      void manifoldL1UnitBallWalk(FeatureGridMDP * gmdp, double step, int num_steps);
      void manifoldL1UnitBallWalkAllSteps(FeatureGridMDP * gmdp, double step);


      double* posteriors = nullptr;
      FeatureGridMDP* MAPmdp = nullptr;
      double MAPposterior;
      
   public:
   
     FeatureGridMDP* mdp = nullptr; //original MDP 
     FeatureGridMDP** R_chain = nullptr; //storing the rewards along the way
      
     ~FeatureBREX(){
        if(R_chain != nullptr) {
          for(unsigned int i=0; i<chain_length; i++) delete R_chain[i];
          delete []R_chain;
        }
        if(posteriors != nullptr) delete []posteriors;
        delete MAPmdp;
        
     }
      
     double getAlpha(){return alpha;}
     
     FeatureBREX(FeatureGridMDP* init_mdp, double min_reward, double max_reward, unsigned int chain_len, double step, double conf, int samp_flag=0, bool reject=false, int num_step=1):  r_min(min_reward), r_max(max_reward), step_size(step), chain_length(chain_len), alpha(conf), sampling_flag(samp_flag), mcmc_reject(reject), num_steps(num_step){ 
     
        unsigned int grid_height = init_mdp -> getGridHeight();
        unsigned int grid_width = init_mdp -> getGridWidth();
        bool* initStates = init_mdp -> getInitialStates();
        bool* termStates = init_mdp -> getTerminalStates();
        nfeatures = init_mdp -> getNumFeatures();
        double* fweights = init_mdp -> getFeatureWeights();
        stateFeatures = init_mdp -> getStateFeatures();
        bool stochastic = init_mdp -> isStochastic();
        gamma = init_mdp -> getDiscount();
        //copy init_mdp
        mdp = new FeatureGridMDP(grid_width, grid_height, initStates, termStates, nfeatures, fweights, stateFeatures, stochastic, gamma);
        mdp->setWallStates(init_mdp->getWallStates());
        initializeMDP(); //set weights to (r_min+r_max)/2
        
        
        MAPmdp = new FeatureGridMDP(grid_width, grid_height, initStates, termStates, nfeatures, fweights, stateFeatures, stochastic, gamma);
        MAPmdp->setWallStates(init_mdp->getWallStates());
        
        
        
        MAPmdp->setFeatureWeights(mdp->getFeatureWeights());
        MAPposterior = 0;
        
        R_chain = new FeatureGridMDP*[chain_length];
        posteriors = new double[chain_length];    
        iteration = 0;
        
       }; 
       
      FeatureGridMDP* getMAPmdp(){return MAPmdp;}
      double getMAPposterior(){return MAPposterior;}
      void run(double eps=0.001);
      double getMinReward(){return r_min;};
      double getMaxReward(){return r_max;};
      double getStepSize(){return step_size;};
      unsigned int getChainLength(){return chain_length;};
      FeatureGridMDP** getRewardChain(){ return R_chain; };
      FeatureGridMDP* getMeanMDP(int burn, int skip);
      double* getPosteriorChain(){ return posteriors; };
      FeatureGridMDP* getMDP(){ return mdp;};
      double calculatePosterior(FeatureGridMDP* gmdp);
      double logsumexp(double* nums, unsigned int size);
      //accumulate the feature counts of a vector
      vector<double> computeFCounts(vector<pair<unsigned int, unsigned int> > traj);
      void addTrajectories(vector<vector<pair<unsigned int,unsigned int> > > demonstrations);
      void addPairwisePreferences(vector<pair<unsigned int,unsigned int> > prefs);

           
};

void FeatureBREX::run(double eps)
{
   
     //cout.precision(10);
    //cout << "itr: " << iteration << endl;
    //clear out previous values if they exist
    if(iteration > 0) for(unsigned int i=0; i<chain_length-1; i++) delete R_chain[i];
    iteration++;
    MAPposterior = 0;
    R_chain[0] = mdp; // so that it can be deleted with R_chain!!!!
    //vector<unsigned int> policy (mdp->getNumStates());
    //cout << "testing" << endl;
    //mdp->valueIteration(eps);//deterministicPolicyIteration(policy);
    //cout << "value iter" << endl;
    //mdp->calculateQValues();
    mdp->displayFeatureWeights();
    double posterior = calculatePosterior(mdp);
    //cout << "init posterior: " << posterior << endl;
    posteriors[0] = exp(posterior); 
    int reject_cnt = 0;
    //BREX iterations 
    for(unsigned int itr=1; itr < chain_length; itr++)
    {
      //cout << "itr: " << itr << endl;
      FeatureGridMDP* temp_mdp = new FeatureGridMDP (mdp->getGridWidth(),mdp->getGridHeight(), mdp->getInitialStates(), mdp->getTerminalStates(), mdp->getNumFeatures(), mdp->getFeatureWeights(), mdp->getStateFeatures(), mdp->isStochastic(), mdp->getDiscount());
      //set the walls
      temp_mdp->setWallStates(mdp->getWallStates());
      
      temp_mdp->setFeatureWeights(mdp->getFeatureWeights());
      if(sampling_flag == 0)
      {   //random grid walk
          modifyFeatureWeightsRandomly(temp_mdp,step_size);
      }
      else if(sampling_flag == 1)
      {
          //cout << "sampling randomly from L1 unit ball" << endl;
          sampleL1UnitBallRandomly(temp_mdp);  
      }
      //updown sampling on L1 ball
      else if(sampling_flag == 2)
      { 
          //cout << "before step" << endl;
          //temp_mdp->displayFeatureWeights();
          updownL1UnitBallWalk(temp_mdp, step_size);
          //cout << "after step" << endl;
          //temp_mdp->displayFeatureWeights();
          //check if norm is right
          assert(isEqual(l1_norm(temp_mdp->getFeatureWeights(), temp_mdp->getNumFeatures()),1.0));
      }
      //random manifold walk sampling
      else if(sampling_flag == 3)
      {
          manifoldL1UnitBallWalk(temp_mdp, step_size, num_steps);
          assert(isEqual(l1_norm(temp_mdp->getFeatureWeights(), temp_mdp->getNumFeatures()),1.0));
      }
      else if(sampling_flag == 4)
      {
          manifoldL1UnitBallWalkAllSteps(temp_mdp, step_size);
          assert(isEqual(l1_norm(temp_mdp->getFeatureWeights(), temp_mdp->getNumFeatures()),1.0));
      }
      //cout << "trying out" << endl;    
      //temp_mdp->displayFeatureWeights();
      
      //temp_mdp->valueIteration(eps, mdp->getValues());
      //temp_mdp->deterministicPolicyIteration(policy);//valueIteration(0.05);
      //temp_mdp->calculateQValues();
      
      double new_posterior = calculatePosterior(temp_mdp);
      //cout << "nwe posterior: " << new_posterior << endl;
      double probability = min((double)1.0, exp(new_posterior - posterior));
      //cout << "probability accept = " << probability << endl;

      //transition with probability
      double r = ((double) rand() / (RAND_MAX));
      if ( r < probability ) //policy_changed && 
      {
         //temp_mdp->displayFeatureWeights();
         //cout << "accept" << endl;
         mdp = temp_mdp;
         posterior = new_posterior;
         R_chain[itr] = temp_mdp;
         posteriors[itr] = exp(new_posterior);
         //if (itr%100 == 0) cout << itr << ": " << posteriors[itr] << endl;
         if(posteriors[itr] > MAPposterior)
         {
           cout << "iter " << itr << endl;
           cout << "new MAP" << endl;
           temp_mdp->displayFeatureWeights();

           MAPposterior = posteriors[itr];
           //TODO remove set terminals, right? why here in first place?
           MAPmdp->setFeatureWeights(mdp->getFeatureWeights());
         }
      }else {
        //delete temp_mdp
        delete temp_mdp;
         //keep previous reward in chain
         //cout << "reject!!!!" << endl;
         reject_cnt++;
         if(mcmc_reject)
         {
            //TODO can I make this more efficient by adding a count variable?
            //make a copy of mdp
            FeatureGridMDP* mdp_copy = new FeatureGridMDP (mdp->getGridWidth(),mdp->getGridHeight(), mdp->getInitialStates(), mdp->getTerminalStates(), mdp->getNumFeatures(), mdp->getFeatureWeights(), mdp->getStateFeatures(), mdp->isStochastic(), mdp->getDiscount());
            //mdp_copy->setValues(mdp->getValues());
            //mdp_copy->setQValues(mdp->getQValues());
            mdp_copy->setWallStates(mdp->getWallStates());
            R_chain[itr] = mdp_copy;
         }
         //sample until you get accept and then add that -- doesn't repeat old reward in chain
         else
         {
             
             
             
             assert(reject_cnt < 100000);
             itr--;
             //delete temp_mdp;
         }
      }
      
    
    }
    cout << "accepts / total: " << chain_length - reject_cnt << "/" << chain_length << endl;
  
}
//optimized version
double FeatureBREX::logsumexp(double* nums, unsigned int size) {
  double max_exp = nums[0];
  double sum = 0.0;
  unsigned int i;
  //find max exponent
  for (i = 1 ; i < size ; i++)
  {
    if (nums[i] > max_exp)
      max_exp = nums[i];
  }

  for (i = 0; i < size ; i++)
    sum += exp(nums[i] - max_exp);

  return log(sum) + max_exp;
}


//computes posterior using pairwise preference softmax likelihood fn
double FeatureBREX::calculatePosterior(FeatureGridMDP* gmdp) //assuming uniform prior
{
    
    double posterior = 0;
    //add in a zero norm (non-zero count)
    double prior = 0;
//    int count = 0;
//    double* weights = gmdp->getFeatureWeights();
//    for(int i=0; i < gmdp->getNumFeatures(); i++)
//        if(abs(weights[i]) > 0.0001)
//            count += 1;
//    prior = -1 * alpha * log(count-1);
    
    posterior += prior;

    double* weights = gmdp->getFeatureWeights();
    
    // "-- Ranked Demos --" each element is a pair of trajectory indices
   for(unsigned int i=0; i < pairwise_preferences.size(); i++)
   {
      pair<unsigned int,unsigned int> trajpair = pairwise_preferences[i];
      unsigned int worse_idx =  trajpair.first;
      unsigned int better_idx = trajpair.second; 
      //cout << "prefrence: " << worse_idx << " < " << better_idx << endl;
      vector<double> fcounts_better = fcounts[better_idx];
      vector<double> fcounts_worse = fcounts[worse_idx];
      //compute dot products
      double better_return = dotProduct(weights, &fcounts_better[0], nfeatures);
      double worse_return = dotProduct(weights, &fcounts_worse[0], nfeatures);
      //cout << "better return = " << better_return << "  worse return = " << worse_return << endl; 
      double Z [2];
      Z[0] = alpha * better_return;
      Z[1] = alpha * worse_return;
      //cout << Z[0] << "," << Z[1] << endl;
      float pairwise_likelihood = alpha * better_return - logsumexp(Z, 2);
      //cout << alpha * better_return << endl;
      //cout << logsumexp(Z,2) << endl;
      //cout << worse_idx << " < " << better_idx << "  loglikelihood = " << pairwise_likelihood << endl;
      posterior += pairwise_likelihood;
      //cout << state << "," << action << ": " << posterior << endl;
   }
   
  
   return posterior;
}

void FeatureBREX::modifyFeatureWeightsRandomly(FeatureGridMDP * gmdp, double step)
{
   unsigned int state = rand() % gmdp->getNumFeatures();
   double change = pow(-1,rand()%2)*step;
   //cout << "before " << gmdp->getReward(state) << endl;
   //cout << "change " << change << endl;
   double weight = max(min(gmdp->getWeight(state) + change, r_max), r_min);
   //if(gmdp->isTerminalState(state)) reward = max(min(gmdp->getReward(state) + change, r_max), 0.0);
   //else reward = max(min(gmdp->getReward(state) + change, 0.0), r_min); 
   //cout << "after " << reward << endl;
   gmdp->setFeatureWeight(state, weight);
}

void FeatureBREX::sampleL1UnitBallRandomly(FeatureGridMDP * gmdp)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = sample_unit_L1_norm(numFeatures);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}

void FeatureBREX::updownL1UnitBallWalk(FeatureGridMDP * gmdp, double step)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = updown_l1_norm_walk(gmdp->getFeatureWeights(), numFeatures, step);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}


void FeatureBREX::manifoldL1UnitBallWalk(FeatureGridMDP * gmdp, double step, int num_steps)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = random_manifold_l1_step(gmdp->getFeatureWeights(), numFeatures, step, num_steps);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}

void FeatureBREX::manifoldL1UnitBallWalkAllSteps(FeatureGridMDP * gmdp, double step)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = take_all_manifold_l1_steps(gmdp->getFeatureWeights(), numFeatures, step);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}

//accumulate the feature counts of a vector
vector<double> FeatureBREX::computeFCounts(vector<pair<unsigned int, unsigned int> > traj)
{
  vector<double> fcounts(nfeatures);
  for(unsigned int t = 0; t < traj.size(); t++)
  {
    pair<unsigned int, unsigned int> p = traj[t];
    unsigned int state = p.first;
    //get feature vector for state
    double* f = stateFeatures[state];
    for(unsigned int i=0; i<nfeatures; i++)
      fcounts[i] += pow(gamma, t) * f[i];
  }
  return fcounts;
}

//compute fcounts for each demo
void FeatureBREX::addTrajectories(vector<vector<pair<unsigned int,unsigned int> > > demonstrations)
{
  for(vector<pair<unsigned int, unsigned int> > traj : demonstrations)
  {
    vector<double> fcs = computeFCounts(traj);
    fcounts.push_back(fcs);
  }

  // for(unsigned int t = 0; t < fcounts.size(); t++)
  // {
  //   cout << "fcounts " << t << endl;
  //   for(unsigned int i = 0;  i < fcounts[t].size(); i++)
  //     cout << fcounts[t][i] << ",";
  //   cout << endl;
  // }

}

//input is a list of pairs (i,j) where j is preferred over i.
void FeatureBREX::addPairwisePreferences(vector<pair<unsigned int,unsigned int> > prefs)
{
  for(pair<unsigned int, unsigned int> p : prefs)
    pairwise_preferences.push_back(p);

  // cout <<"preferences" << endl;
  // for(pair<unsigned int, unsigned int> p : pairwise_preferences)
  //   cout << "(" << p.first << ", " << p.second << ")" << endl;

}

void FeatureBREX::initializeMDP()
{
//   if(sampling_flag == 0)
//   {
//       double* weights = new double[mdp->getNumFeatures()];
//       for(unsigned int s=0; s<mdp->getNumFeatures(); s++)
//       {
//          weights[s] = (r_min+r_max)/2;
//       }
//       mdp->setFeatureWeights(weights);
//       delete [] weights;
//   }
//   else if (sampling_flag == 1)  //sample randomly from L1 unit ball
//   {
//       double* weights = sample_unit_L1_norm(mdp->getNumFeatures());
//       mdp->setFeatureWeights(weights);
//       delete [] weights;
//   }    
//   else if(sampling_flag == 2)
//   {
       unsigned int numDims = mdp->getNumFeatures();
       double* weights = new double[numDims];
       for(unsigned int s=0; s<numDims; s++)
           weights[s] = -1.0 / numDims;
//       {
//          if((rand() % 2) == 0)
//            weights[s] = 1.0 / numDims;
//          else
//            weights[s] = -1.0 / numDims;
////            if(s == 0)
////                weights[s] = 1.0;
////            else
////                weights[s] = 0.0;
//       }
//       weights[0] = 0.2;
//       weights[1] = 0.2;
//       weights[2] = -0.2;
//       weights[3] = 0.2;
//       weights[4] = 0.2;
       //weights[0] = 1.0;
       mdp->setFeatureWeights(weights);
       delete [] weights;
//   }
//   else if(sampling_flag == 3)
//   {
//       unsigned int numDims = mdp->getNumFeatures();
//       double* weights = new double[numDims];
//       for(unsigned int s=0; s<numDims; s++)
//           weights[s] = 0.0;
////       {
////          if((rand() % 2) == 0)
////            weights[s] = 1.0 / numDims;
////          else
////            weights[s] = -1.0 / numDims;
//////            if(s == 0)
//////                weights[s] = 1.0;
//////            else
//////                weights[s] = 0.0;
////       }
////       weights[0] = 0.2;
////       weights[1] = 0.2;
////       weights[2] = -0.2;
////       weights[3] = 0.2;
////       weights[4] = 0.2;
//       weights[0] = 1.0;
//       mdp->setFeatureWeights(weights);
//       delete [] weights;
//   }

}



FeatureGridMDP* FeatureBREX::getMeanMDP(int burn, int skip)
{
    //average rewards in chain
    int nFeatures = mdp->getNumFeatures();
    double aveWeights[nFeatures];
    
    for(int i=0;i<nFeatures;i++) aveWeights[i] = 0;
    
    int count = 0;
    for(unsigned int i=burn; i<chain_length; i+=skip)
    {
        count++;
        //(*(R_chain + i))->displayFeatureWeights();
        //cout << "weights" << endl;
        double* w = (*(R_chain + i))->getFeatureWeights();
        for(int f=0; f < nFeatures; f++)
            aveWeights[f] += w[f];
        
    }
    for(int f=0; f < nFeatures; f++)
        aveWeights[f] /= count;
  
//    //create new MDP with average weights as features
    FeatureGridMDP* mean_mdp = new FeatureGridMDP(MAPmdp->getGridWidth(),MAPmdp->getGridHeight(), MAPmdp->getInitialStates(), MAPmdp->getTerminalStates(), MAPmdp->getNumFeatures(), aveWeights, MAPmdp->getStateFeatures(), MAPmdp->isStochastic(), MAPmdp->getDiscount());
    mean_mdp->setWallStates(MAPmdp->getWallStates());
    
    
    return mean_mdp;
}


#endif
