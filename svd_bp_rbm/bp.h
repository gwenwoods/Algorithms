#ifndef BP_H
#define BP_H

#include <string>

#include "abstractAlgorithm.h"

using namespace std;

class BP : public abstractAlgorithm
{
public:
        BP(int inputUnitNum, int hiddenUnitNum, int outputUnitNum, int dataNum);
        ~BP();
        void run(int epochNum, float lrate);
        void loadTrainingData(string INITIAL_TRAINING_FILE, string TARGET_FILE);
        void recall(string RECALL_FILE);
        float* recall(float* recallInput); 
protected:
        inline void updateWeight(int record);
        inline float sgm(float x);

private:
        float *inputUnit;
        float *hiddenUnit;
        float *outputUnit;

        float **weight_InputHidden;
        float **weight_HiddenOutput;
        float **d_weight_InputHidden;
        float **d_weight_HiddenOutput;

        float **data;
        float **target;

        int INPUT_UNIT_NUM;
        int HIDDEN_UNIT_NUM;
        int OUTPUT_UNIT_NUM;
        int DATA_NUM;
 
        float lrate;
};

#endif
