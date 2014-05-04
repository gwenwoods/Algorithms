#include <iostream>
#include <fstream>
#include <sstream>

#include "svd.h"


#define ZERO 0
#define COMMA ','

using namespace std;

SVD::SVD(int itemNum, int customerNum, int dataNum)
{
    itemFeatureValue = new float[itemNum];
    customerFeatureValue = new float[customerNum];
    data_customerId = new unsigned int[dataNum];
    data_itemId = new unsigned int[dataNum];
    data_rate = new unsigned int[dataNum];
    data_residual = new float[dataNum];

    averageItemRating = new float[itemNum];
    averageCustomerOffset = new float[customerNum];

    DATA_NUM = dataNum;
    CUSTOMER_NUM = customerNum;
    ITEM_NUM = itemNum;
    
    maxRating = 1000000;
    minRating = -1000000;
}

SVD::~SVD()
{
    delete [] itemFeatureValue;
    delete [] customerFeatureValue;
    delete [] data_customerId;
    delete [] data_itemId;
    delete [] data_rate;
    delete [] data_residual;

    delete [] averageItemRating;
    delete [] averageCustomerOffset;
}

void SVD::setRatingBoundry(float max, float min)
{
    maxRating = max;
    minRating = min;
}

void SVD::loadInitialSetting(string INITIAL_ITEM_MEAN_FILE, string INITIAL_CUSTOMER_OFFSET_FILE, 
                         float ITEM_FEATURE_INITIAL_VALUE, float CUSTOMER_FEATURE_INITIAL_VALUE)
{
   itemFeatureInitialValue = ITEM_FEATURE_INITIAL_VALUE;
   customerFeatureInitialValue = CUSTOMER_FEATURE_INITIAL_VALUE;

   cout << currentTime() << "   start reading initial setting data ...  " << endl;
   ifstream inputItem(INITIAL_ITEM_MEAN_FILE.c_str());
   for(int i=ZERO; i< ITEM_NUM; i++) {
    string tmp;
    getline(inputItem,tmp, COMMA);
    int item_id = atoi(tmp.c_str());
    getline(inputItem,tmp,COMMA);
    float avg = atof(tmp.c_str());
    getline(inputItem,tmp);
    averageItemRating[item_id] = avg;
    itemFeatureValue[item_id] = ITEM_FEATURE_INITIAL_VALUE; // this line is to initiate intmFeatureValue for one featur
   }
   inputItem.close();

   ifstream inputUserOffset(INITIAL_CUSTOMER_OFFSET_FILE.c_str());
   for(int i= ZERO; i< CUSTOMER_NUM; i++) {
    string tmp;
    getline(inputUserOffset,tmp,COMMA);
    int cid = atoi(tmp.c_str());
    getline(inputUserOffset,tmp,COMMA);
    float offset = atof(tmp.c_str());
    getline(inputUserOffset,tmp);
    averageCustomerOffset[cid] = offset;
    customerFeatureValue[cid] = CUSTOMER_FEATURE_INITIAL_VALUE; // this line is to initiate userValue for one feature
  }
  inputUserOffset.close();

  cout << currentTime() << "   finish reading initial setting data ...  " << endl;

}



void SVD::loadTrainingData(string INITIAL_TRAINING_FILE, float DATA_RESIDUAL_INITIAL_VALUE)
{
  cout << currentTime() << "  read training data set ...  " << endl;
  ifstream input(INITIAL_TRAINING_FILE.c_str());

  for (int i=ZERO; i < DATA_NUM; i++){
    string tmp;
    getline(input,tmp,COMMA);
    int cid = atoi(tmp.c_str());
    getline(input,tmp,COMMA);
    int mid = atoi(tmp.c_str());
    getline(input,tmp,COMMA);
    int rate = atoi(tmp.c_str());
    getline(input,tmp);

    data_customerId[i] = cid;
    data_itemId[i] = mid;
    data_rate[i] = rate;
    data_residual[i] = DATA_RESIDUAL_INITIAL_VALUE;
    if(i%10000000 == ZERO) cout << i << endl;
  }
  input.close();
  cout << currentTime() << "   finish reading training data ...  " << endl;
}

void SVD::run(int featureNum, int epochNum, float lrate, float Kvalue)
{
 this ->lrate = lrate;
 this ->Kvalue = Kvalue;

// cout << "lrate = " << this -> lrate << endl;
// cout << "Kvalue = " << this -> Kvalue << endl;

 for(int currentFeature = ZERO; currentFeature < featureNum; currentFeature++) {
    stringstream featureNumString;
    featureNumString << currentFeature;
    for(int epoch = ZERO; epoch < epochNum ; epoch++){
      cout << currentTime() << "   start to train epoch " << epoch << " ... " << endl;
      for (int i= ZERO; i< DATA_NUM; i++ ) {
        if (currentFeature == ZERO && epoch == ZERO) {train_Baseline(data_itemId[i], data_customerId[i], data_rate[i]);}
        else {
            train(data_itemId[i], data_customerId[i], data_rate[i], i);
        }
      }
      cout << currentTime() << "   finish training epoch " << epoch << " ... " << endl;
    }

    //--------------------------------
    // update residual
    cout << currentTime() << "   start to update residual ... " << endl;
    for (int i=ZERO; i< DATA_NUM; i++ ) {
      data_residual[i] =  updateResidual(data_itemId[i], data_customerId[i], i);
    }
    cout << currentTime() << "   finish updating residual ... " << endl;
    //----------------------------
    //output result: Movie Feature coefficient
    cout << currentTime() << "   start to output item feature coefficients ... " << currentFeature << endl;
    char itemFeatureFile[80];
    strcpy(itemFeatureFile,"./itemFeature_");
    strcat(itemFeatureFile, featureNumString.str().c_str());
    strcat(itemFeatureFile,".csv");
    ofstream outputItem(itemFeatureFile);
    for(int i=ZERO; i< ITEM_NUM; i++) {
      outputItem << itemFeatureValue[i] << COMMA << endl;
      itemFeatureValue[i] = 0.1;
    }
    outputItem.close();
    //----------------------------------
    //output result: User Feature coeffiecient
    cout << currentTime() << "   start to output user feature coefficients ... " << currentFeature << endl;
    char userFeatureFile[80];
    strcpy(userFeatureFile,"./userFeature_");
    strcat(userFeatureFile, featureNumString.str().c_str());
    strcat(userFeatureFile,".csv");
    ofstream outputUser(userFeatureFile);
    for(int i=0; i< CUSTOMER_NUM; i++) {
      outputUser << customerFeatureValue[i] << COMMA << endl;
      customerFeatureValue[i] = 0.1;
    }
    outputUser.close();
    //-----------------------------------------
    //output result: Residual after current feature
    if( currentFeature > 0 && currentFeature % 50 == 0) {
      cout << currentTime() << "   start to output residuals ... " << currentFeature << endl;
      char residualFile[80];
      strcpy(residualFile,"./residual_");
      strcat(residualFile, featureNumString.str().c_str());
      strcat(residualFile,".csv");
           ofstream outputResidual(residualFile);
      for(int i= ZERO; i<DATA_NUM; i++){
        outputResidual << data_residual[i]<< COMMA <<endl;
      }
      outputResidual.close();
    }
    //---------------------------------------------------------
  } // feature loop
  cout << currentTime() << "   code finished ... " << endl;

}

float SVD::predictRating_Baseline(int item, int customer)
{
  float predict = averageItemRating[item] + averageCustomerOffset[customer];
  if(predict > maxRating) {predict = maxRating;}
  if(predict < minRating) {predict = minRating;}

  return predict;
}

float SVD::predictRating(int item, int customer, int data_id)
{
  float predict = ZERO;
  predict = data_residual[data_id] + customerFeatureValue[customer]*itemFeatureValue[item];
  if(predict > maxRating) {predict = maxRating;}
  if(predict < minRating) {predict = minRating;}
  return predict;
}

void SVD::train_Baseline(int item, int customer, float rating)
{
  float err = rating - predictRating_Baseline(item, customer);
  float uv = customerFeatureValue[customer];
  customerFeatureValue[customer] += lrate * (err* itemFeatureValue[item] - Kvalue * customerFeatureValue[customer]);
  itemFeatureValue[item] += lrate * (err * uv - Kvalue * itemFeatureValue[item]);
}

void SVD::train(int item, int customer, float rating, int data_id)
{
  float err = rating - predictRating(item, customer, data_id);
  float uv = customerFeatureValue[customer];
  customerFeatureValue[customer] += lrate * (err* itemFeatureValue[item] - Kvalue * customerFeatureValue[customer]);
  itemFeatureValue[item] += lrate * (err * uv - Kvalue * itemFeatureValue[item]);
}

float SVD::updateResidual(int item, int customer, int data_id)
{
  float value = data_residual[data_id] + customerFeatureValue[customer]*itemFeatureValue[item];
  if(value > maxRating) {value = maxRating;}
  if(value < minRating) {value = minRating;}
  return value;
}

