//-----------------------------------------------------------------
//  RBM: Restricted Boltzmann Machine
//
//  RBM is a special kind of Boltzmann machine, which does not have
//  connections between visible-visible units and hidden-hidden units.
//  
//
//  weight[i][j]: weight connection between the visible unit i and 
//                the hidden unit j
//  biasV[i]: bias for the visible unit i
//  biasH[j]: bias for the hidden unit j
//  lrate: learning rate
//
//                 
//  Written by: Wen-Ching Lin 
//  Date: Aug 2007

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include "rbm.h"

#define COMMA ','

using namespace std;

RBM::RBM(int visibleUnitNum, int hiddenUnitNum, int recordNum)
{

    weight = new float*[visibleUnitNum];
    d_weight = new float*[visibleUnitNum];
    for(int i=ZERO; i< visibleUnitNum; i++){
       weight[i] = new float[hiddenUnitNum];
       d_weight[i] = new float[hiddenUnitNum]; 
    }

    biasV = new float[visibleUnitNum];
    biasH = new float[hiddenUnitNum];

    pvu_neg = new float*[recordNum];
    recordStateV_pos = new float*[recordNum];
    recordStateV_neg = new float*[recordNum];
    for(int i = ZERO; i < recordNum; i++ ){
       pvu_neg[i] = new float[visibleUnitNum]; 
       recordStateV_pos[i] = new float[visibleUnitNum];
       recordStateV_neg[i] = new float[visibleUnitNum];
    }

    phu_pos = new float*[recordNum];
    phu_neg = new float*[recordNum];
    recordStateH_pos = new float*[recordNum];
    recordStateH_neg = new float*[recordNum];
    for(int i = ZERO; i < recordNum; i++ ){
       phu_pos[i] = new float[hiddenUnitNum];
       phu_neg[i] = new float[hiddenUnitNum];
       recordStateH_pos[i] = new float[hiddenUnitNum];
       recordStateH_neg[i] = new float[hiddenUnitNum];
    }

     VISIBLE_UNIT_NUM = visibleUnitNum;
    HIDDEN_UNIT_NUM = hiddenUnitNum;
    RECORD_NUM = recordNum;

     for (int i = 0; i < VISIBLE_UNIT_NUM; i++) {
        for (int j = 0; j < HIDDEN_UNIT_NUM; j++) {
            weight[i][j] = 0.1 * (rand() % 100 );
            d_weight[i][j] = 0.0;
        }
    }

    for(int i = 0 ; i < VISIBLE_UNIT_NUM ; i++) {
        for(int record = 0 ; record<RECORD_NUM; record++) {
            pvu_neg[record][i] = 0.0;
        }
        biasV[i] = 0;
    }

    for(int i = 0 ; i < HIDDEN_UNIT_NUM ; i++) {
        for(int record= 0; record>RECORD_NUM; record++) {
            phu_pos[record][i] = 0.0;
            phu_neg[record][i] = 0.0 ;
        }
        biasH[i] = 0;
    }

}

RBM::~RBM()
{
    for(int i=ZERO; i< VISIBLE_UNIT_NUM; i++){
       delete [] weight[i];
       delete [] d_weight[i];
    }

    for(int i=ZERO; i< RECORD_NUM; i++) {
       delete [] pvu_neg[i];
       delete [] phu_pos[i];
       delete [] phu_neg[i];
       delete [] recordStateV_pos[i];
       delete [] recordStateV_neg[i];
       delete [] recordStateH_pos[i];
       delete [] recordStateH_neg[i];
    }
 
    delete [] weight;
    delete [] d_weight;
    delete [] biasV;
    delete [] biasH;
}

//void RBM::loadTrainingData(string INITIAL_TRAINING_FILE, float DATA_RESIDUAL_INITIAL_VALUE){}

void RBM::loadTrainingData(string INITIAL_TRAINING_FILE)
{
  cout << currentTime() << "  read training data set ...  " << endl;
  ifstream input(INITIAL_TRAINING_FILE.c_str());

  for (int record=ZERO; record < RECORD_NUM; record++){
     for(int i = ZERO ; i<VISIBLE_UNIT_NUM; i++) { 
        input >> recordStateV_pos[record][i];
     }
  }
  input.close();
  cout << currentTime() << "   finish reading training data ...  " << endl;
}

void RBM::run(int epochNum, float lrate)
{
    this ->lrate = lrate;
    float T = 1;
    for(int epoch = ZERO; epoch < epochNum ; epoch++){
      if(epoch%100 == 0){
        cout << currentTime() << "   start to train epoch " << epoch << " ... " << endl;
      }

      for (int i= ZERO; i< RECORD_NUM; i++ ) {
        updatePHU_pos(i, T);
        updatePVU_neg(i, T);
        updatePHU_neg(i, T);
      }
      updateWeight();
      if(epoch%100 == 0){
        cout << currentTime() << "   finish training epoch " << epoch << " ... " << endl;
      }
    }
}


void RBM::updatePVU_neg(int record, float T)
{
    float beta = 1.0/T;
    for(int i = 0; i < VISIBLE_UNIT_NUM; i++ ){
        float ha = 0;
        for(int j = 0; j < HIDDEN_UNIT_NUM; j++) {
            ha += weight[i][j] * recordStateH_pos[record][j];
        }
        ha += biasV[i];
        pvu_neg[record][i] = 1/(1+exp(-1*beta*ha));
    }
    for( int i = 0; i < VISIBLE_UNIT_NUM; i++ ){
        float a = 0.01*(rand()%100);
        if(a <= pvu_neg[record][i]){ recordStateV_neg[record][i] = 1 ; }
        else { recordStateV_neg[record][i] = 0;}
    }
}

void RBM::updatePHU_pos(int record, float T)
{
    float beta = 1.0/T;
    for(int j = 0; j < HIDDEN_UNIT_NUM; j++ ){
        float ha = 0;
        for(int i = 0; i < VISIBLE_UNIT_NUM; i++) {
            ha += weight[i][j] * recordStateV_pos[record][i];
        }
        ha += biasH[j];
        phu_pos[record][j] = 1/(1+exp(-1*beta*ha));
    }

    for( int j = 0; j < HIDDEN_UNIT_NUM; j++ ){
        float a = 0.01* (rand()%100);
        if(a <= phu_pos[record][j]){ recordStateH_pos[record][j] = 1 ; }
        else { recordStateH_pos[record][j] = 0;}
    }

}

void RBM::updatePHU_neg(int record, float T)
{
    float beta = 1.0/T;
    for(int j = 0; j < HIDDEN_UNIT_NUM; j++ ){
        float ha = 0;
        for(int i = 0; i < VISIBLE_UNIT_NUM; i++) {
            ha += weight[i][j] * pvu_neg[record][i];
        }
        ha += biasH[j];
        phu_neg[record][j] = 1/(1+exp(-1*beta*ha));
    }

    for( int j = 0; j < HIDDEN_UNIT_NUM; j++ ){
        float a = 0.01* (rand()%100);
        if(a <= phu_neg[record][j]){ recordStateH_neg[record][j] = 1 ; }
        else { recordStateH_neg[record][j] = 0;}
    }

}

void RBM::updateWeight()
{
       for(int i = 0; i < VISIBLE_UNIT_NUM; i++) {
        for(int j=0; j< HIDDEN_UNIT_NUM; j++){
            float clamp = 0;
            float freeRun = 0;
            for(int record = 0; record < RECORD_NUM; record++) {
                clamp += (float)recordStateV_pos[record][i]*phu_pos[record][j];
                freeRun += (float)pvu_neg[record][i]*phu_neg[record][j];
            }
            clamp = clamp/(float)RECORD_NUM;
            freeRun = freeRun/(float)RECORD_NUM;
            d_weight[i][j] = /*d_weight[i][j]*momentum +*/ lrate *(clamp - freeRun);
            weight[i][j] = weight[i][j] + d_weight[i][j];//  - 0.0002 * weight[i][j] ;
        }
    }

  //  cout << "V0 " << weight[0][0] << " " << weight[0][1] << " " << weight[0][2] << " "  << weight[0][3] <<  " " << biasV[0] << endl;
  //  cout << "V1 " << weight[1][0] << " " << weight[1][1] << " " << weight[1][2] << " "  << weight[1][3] <<  " " << biasV[1] << endl;
  //  cout << "V2 " << weight[2][0] << " " << weight[2][1] << " " << weight[2][2] << " "  << weight[2][3] <<  " " << biasV[2] << endl;
  //  cout << "BH " << biasH[0]     << " " << biasH[1]     << " " << biasH[2]     << " "  << biasH[3]     << endl << endl;

    for(int i = 0; i < VISIBLE_UNIT_NUM; i++) {
        float clamp = 0;
        float freeRun = 0;
        for(int record = 0; record < RECORD_NUM; record++) {
            clamp += (float)recordStateV_pos[record][i];
            freeRun += (float)pvu_neg[record][i];
        }
        clamp = clamp/(float)RECORD_NUM;
        freeRun = freeRun/(float)RECORD_NUM;
        biasV[i]+= lrate *(clamp - freeRun) ;
    }

    for(int j=0; j< HIDDEN_UNIT_NUM; j++){
        float clamp = 0;
        float freeRun = 0;
        for(int record = 0; record < RECORD_NUM; record++) {
            clamp += (float)phu_pos[record][j];
            freeRun += (float)phu_neg[record][j];
        }
        clamp = clamp/(float)RECORD_NUM;
        freeRun = freeRun/(float)RECORD_NUM;
        biasH[j] += lrate *(clamp - freeRun) ;
    }

}

float** RBM::getRecordStateV_neg(){
   return recordStateV_neg;
}
