//-----------------------------------------------------------------
//  BP: Back Propagation
//
//  lrate: learning rate
//                 
//  Written by: Wen-Ching Lin 
//  Date: Jan 2007  (v1.0)
//        Aug 2007  (first OO version)

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include "bp.h"

#define COMMA ','

using namespace std;

BP::BP(int inputUnitNum, int hiddenUnitNum, int outputUnitNum, int dataNum)
{
    inputUnit = new float[inputUnitNum+1]; //input[0] is the bias
    hiddenUnit = new float[hiddenUnitNum+1]; //hidden[0] is the bias 
    outputUnit = new float[outputUnitNum];
 
    weight_InputHidden = new float*[inputUnitNum+1];
    d_weight_InputHidden = new float*[inputUnitNum+1];
    for(int i = 0 ; i <= inputUnitNum; i++){
       weight_InputHidden[i] = new float[hiddenUnitNum+1];
       d_weight_InputHidden[i] = new float[hiddenUnitNum+1];
    }   
 
    weight_HiddenOutput = new float*[hiddenUnitNum+1];
    d_weight_HiddenOutput = new float*[hiddenUnitNum+1];
    for(int i = 0 ; i <= hiddenUnitNum; i++){
       weight_HiddenOutput[i] = new float[outputUnitNum];
       d_weight_HiddenOutput[i] = new float[outputUnitNum];
    }

    data = new float*[dataNum];
    target  = new float*[dataNum];
    for(int record = 0; record < dataNum; record++) {
       data[record] = new float[inputUnitNum+1];
       target[record] = new float[outputUnitNum];
    }


    INPUT_UNIT_NUM = inputUnitNum;
    HIDDEN_UNIT_NUM = hiddenUnitNum;
    OUTPUT_UNIT_NUM = outputUnitNum;
    DATA_NUM = dataNum;
   
     for(int j=0; j<= hiddenUnitNum; j++) {
      for(int i=0; i<= inputUnitNum; i++) {
        weight_InputHidden[i][j] = (i+j)*0.1;
      }
   }

   for(int k=0; k< outputUnitNum; k++) {
     for(int j=0; j<= hiddenUnitNum; j++) {
       weight_HiddenOutput[j][k] = (j+k)*0.1;
     }
   }

}

BP::~BP()
{
    for(int i=ZERO; i<= INPUT_UNIT_NUM; i++){
       delete [] weight_InputHidden[i];
       delete [] d_weight_InputHidden[i];
    }

    for(int i=ZERO; i<= HIDDEN_UNIT_NUM; i++){
       delete [] weight_HiddenOutput[i];
       delete [] d_weight_HiddenOutput[i];
    }
}


void BP::loadTrainingData(string INITIAL_TRAINING_FILE, string TARGET_FILE)
{
  cout << currentTime() << "  read training data set ...  " << endl;
  ifstream inputI(INITIAL_TRAINING_FILE.c_str());
  ifstream inputT(TARGET_FILE.c_str());

  for (int record=ZERO; record < DATA_NUM; record++){
     for(int i = 1 ; i<=INPUT_UNIT_NUM; i++) { 
        inputI >> data[record][i];
     }
     for(int k = 0 ; k< OUTPUT_UNIT_NUM; k++) {
        inputT >> target[record][k];
     }
  }
  inputI.close();
  inputT.close();
  cout << currentTime() << "   finish reading training data ...  " << endl;
}

void BP::run(int epochNum, float lrate)
{
    this ->lrate = lrate;
    for(int epoch = ZERO; epoch < epochNum ; epoch++){
      if(epoch%100 == 0){
        cout << currentTime() << "   start to train epoch " << epoch << " ... " << endl;
      }

      for (int record= ZERO; record< DATA_NUM; record++ ) {
         inputUnit[0] = 1;
         for(int i= 1; i<= INPUT_UNIT_NUM; i++){
             inputUnit[i] = data[record][i]; 
         }
         updateWeight(record);
      }
      if(epoch%100 == 0){
        cout << currentTime() << "   finish training epoch " << epoch << " ... " << endl;
      }
    }
}


void BP::updateWeight(int record)
{
       hiddenUnit[0] = 1;
       for(int j=1; j<=HIDDEN_UNIT_NUM; j++) {
          double aj = 0;
         for(int i = 0; i <= INPUT_UNIT_NUM; i++) {
           aj += weight_InputHidden[i][j] * inputUnit[i];
         }
          hiddenUnit[j] = sgm(aj);
       }
  //----------------------------------
       // Calculate for the output nodes
       for(int k=0; k< OUTPUT_UNIT_NUM; k++) {
         double ak = 0;
         for(int j = 0; j <= HIDDEN_UNIT_NUM; j++) {
           ak += weight_HiddenOutput[j][k]*hiddenUnit[j];
         }
//       O[k] = sgm(ak);
         outputUnit[k] = ak;
       }
       //----------------------------------
       // Calculate gradient
       double deltaOut[OUTPUT_UNIT_NUM];
       double deltaHidden[HIDDEN_UNIT_NUM+1];
       for(int k=0; k< OUTPUT_UNIT_NUM; k++) {
         deltaOut[k] = outputUnit[k]-target[record][k];
       }
       for(int j=0; j<=HIDDEN_UNIT_NUM; j++) {
         double sum = 0;
         for(int k=0; k< OUTPUT_UNIT_NUM; k++) {
           sum += weight_HiddenOutput[j][k]*deltaOut[k];
         }
         deltaHidden[j] = hiddenUnit[j]*(1-hiddenUnit[j])*sum;
       }

       //----------------------------------------
       // update weight
       for(int j=0; j<= HIDDEN_UNIT_NUM; j++) {
         for(int i=0; i<= INPUT_UNIT_NUM; i++) {
           d_weight_InputHidden[i][j] = -1*lrate*deltaHidden[j]*inputUnit[i];
           weight_InputHidden[i][j] += d_weight_InputHidden[i][j];
         }
       }
       for(int k=0; k< OUTPUT_UNIT_NUM; k++) {
         for(int j=0; j<= HIDDEN_UNIT_NUM; j++) {
           d_weight_HiddenOutput[j][k] = -1*lrate*deltaOut[k]*hiddenUnit[j];
           weight_HiddenOutput[j][k] += d_weight_HiddenOutput[j][k];
         }
       }
}

float BP::sgm(float x) {
  double y;
  y = 1/(1+exp(-1*x));
  return y;
}

float* BP::recall(float* recallInput){
    inputUnit[0] = 1;
    for(int i = 1; i <= INPUT_UNIT_NUM; i++) {
        inputUnit[i] = recallInput[i-1];
    }
     hiddenUnit[0] = 1;
       for(int j=1; j<=HIDDEN_UNIT_NUM; j++) {
          double aj = 0;
         for(int i = 0; i <= INPUT_UNIT_NUM; i++) {
           aj += weight_InputHidden[i][j] * inputUnit[i];
         }
          hiddenUnit[j] = sgm(aj);
       }
  //----------------------------------
       // Calculate for the output nodes
       for(int k=0; k< OUTPUT_UNIT_NUM; k++) {
         double ak = 0;
         for(int j = 0; j <= HIDDEN_UNIT_NUM; j++) {
           ak += weight_HiddenOutput[j][k]*hiddenUnit[j];
         }
//       O[k] = sgm(ak);
         outputUnit[k] = ak;
       }
    return outputUnit;
}
