#ifndef RBM_H
#define RBM_H

#include <string>

#include "abstractAlgorithm.h"

using namespace std;

class RBM : public abstractAlgorithm
{
public:
        RBM(int visibleUnitNum, int hiddenUnitNum, int recordNum);
        ~RBM();
        void setRatingBoundry(float max, float min);
        void run(int epochNum, float lrate);
        void loadTrainingData(string INITIAL_TRAINING_FILE, float dataResidualInitialValue);
        void loadTrainingData(string INITIAL_TRAINING_FILE);
        float** getRecordStateV_neg();
       
protected:
        inline void updatePVU_neg(int record, float T);
        inline void updatePHU_pos(int record, float T);
        inline void updatePHU_neg(int record, float T);
        inline void updateWeight();

private:
        float **weight;
        float **d_weight;
        float *biasV;
        float *biasH;

        float **pvu_neg;
        float **phu_pos;
        float **phu_neg;

        float **recordStateV_pos;
        float **recordStateV_neg;
        float **recordStateH_pos;
        float **recordStateH_neg;

        int VISIBLE_UNIT_NUM;
        int HIDDEN_UNIT_NUM;
        int RECORD_NUM;
        float lrate;
};

#endif
