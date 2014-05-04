#ifndef SVD_H
#define SVD_H

#include <string>

#include "abstractAlgorithm.h"

using namespace std;

class SVD : public abstractAlgorithm
{
public:
        SVD(int itemNum, int customerNum, int dataNum);
        ~SVD();
        void setRatingBoundry(float max, float min);
        void run(int featureNum, int epochNum, float lrate, float Kvalue);
        void loadInitialSetting(string INITIAL_ITEM_MEAN_FILE, string INITIAL_CUSTOMER_OFFSET_FILE,
                float itemFeatureInitialValue, float customerFeatureInitialValue);
        void loadTrainingData(string INITIAL_TRAINING_FILE, float dataResidualInitialValue);

protected:
        inline void train_Baseline(int item, int customer, float rating);
        inline void train(int item, int customer, float rating, int data_id);
        inline float updateResidual(int movie, int user, int data_id);
        inline float predictRating_Baseline(int movie, int user);
        inline float predictRating(int movie, int user, int data_id);


private:
        float *itemFeatureValue;
        float *customerFeatureValue;
        float *data_residual;
        unsigned int *data_customerId;
        unsigned int *data_itemId;
        unsigned int *data_rate;

        float *averageItemRating;
        float *averageCustomerOffset; 
        int DATA_NUM;
        int CUSTOMER_NUM;
        int ITEM_NUM;
        float lrate;
        float Kvalue;

        float maxRating;
        float minRating;

        float itemFeatureInitialValue;
        float customerFeatureInitialValue;
        float dataResidualInitialValue;
};

#endif
