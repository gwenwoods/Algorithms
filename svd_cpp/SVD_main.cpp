#include <iostream>
#include <fstream>

#include "abstractAlgorithm.h"
#include "svd.h"

int CUSTOMER_NUM;
int ITEM_NUM;
int DATA_NUM; 
int FEATURE_NUM;
int EPOCH_NUM;

float lrate;
float Kvalue;

float MAX_RATING;
float MIN_RATING;

string INITIAL_TRAINING_FILE;
string INITIAL_ITEM_MEAN_FILE;
string INITIAL_CUSTOMER_OFFSET_FILE;

float ITEM_FEATURE_INITIAL_VALUE;
float CUSTOMER_FEATURE_INITIAL_VALUE;
float DATA_RESIDUAL_INITIAL_VALUE;

using namespace std;

int main(int argc, char *argv[])
{
   char parameterFileName[50];
   strcpy(parameterFileName,"");
   strcat(parameterFileName, argv[1]);
    
   ifstream input(parameterFileName);
   string tmp;
   char *token;
   while(!input.eof()){
      getline(input, tmp);
      if(tmp.length() > 0){
        char *line = strdup(tmp.c_str());
        token = strtok(line," ");
        string paraName ;
        paraName.insert(0,token);
        token = strtok(NULL," ");
        if(paraName == "ITEM_NUM")  {ITEM_NUM = atoi(token); cout << "ITEM_NUM = " << ITEM_NUM << endl;}
        if(paraName == "CUSTOMER_NUM")  {CUSTOMER_NUM = atoi(token); cout << "CUSTOMER_NUM = " << CUSTOMER_NUM << endl;}
        if(paraName == "DATA_NUM")  {DATA_NUM = atoi(token); cout << "DATA_NUM = " << DATA_NUM << endl;}
        if(paraName == "FEATURE_NUM")  {FEATURE_NUM = atoi(token); cout << "FEATURE_NUM = " << FEATURE_NUM << endl;}
        if(paraName == "EPOCH_NUM")  {EPOCH_NUM = atoi(token); cout << "EPOCH_NUM = " << EPOCH_NUM << endl;}

        if(paraName == "lrate")  {lrate = atof(token); cout << "lrate = " << lrate << endl;}
        if(paraName == "Kvalue")  {Kvalue = atof(token); cout << "Kvalue = " << Kvalue << endl;}

        if(paraName == "MAX_RATING")  {MAX_RATING = atof(token); cout << "MAX_RATING = " << MAX_RATING << endl;}
        if(paraName == "MIN_RATING")  {MIN_RATING = atof(token); cout << "MIN_RATING = " << MIN_RATING << endl;}
   
        if(paraName == "INITIAL_TRAINING_FILE")  { INITIAL_TRAINING_FILE.insert(0,token);
                 cout << "INITIAL_TRAINING_FILE = " << INITIAL_TRAINING_FILE << endl;}

        if(paraName == "INITIAL_ITEM_MEAN_FILE")  { INITIAL_ITEM_MEAN_FILE.insert(0,token);
                 cout << "INITIAL_ITEM_MEAN_FILE = " << INITIAL_ITEM_MEAN_FILE << endl;}

        if(paraName == "INITIAL_CUSTOMER_OFFSET_FILE")  { INITIAL_CUSTOMER_OFFSET_FILE.insert(0,token);
                 cout << "INITIAL_CUSTOMER_OFFSET_FILE = " << INITIAL_CUSTOMER_OFFSET_FILE << endl;}

     
        if(paraName == "ITEM_FEATURE_INITIAL_VALUE")  {ITEM_FEATURE_INITIAL_VALUE = atof(token); 
                 cout << "ITEM_FEATURE_INITIAL_VALUE = " << ITEM_FEATURE_INITIAL_VALUE << endl;}
        if(paraName == "CUSTOMER_FEATURE_INITIAL_VALUE")  {CUSTOMER_FEATURE_INITIAL_VALUE = atof(token); 
                 cout << "CUSTOMER_FEATURE_INITIAL_VALUE = " << CUSTOMER_FEATURE_INITIAL_VALUE << endl;}
        if(paraName == "DATA_RESIDUAL_INITIAL_VALUE")  {DATA_RESIDUAL_INITIAL_VALUE = atof(token);
                 cout << "DATA_RESIDUAL_INITIAL_VALUE = " << DATA_RESIDUAL_INITIAL_VALUE << endl;}

      }
    }
   input.close();
  
   SVD svdModel(ITEM_NUM,CUSTOMER_NUM,DATA_NUM); 
   svdModel.setRatingBoundry(MAX_RATING, MIN_RATING);
   svdModel.loadInitialSetting(INITIAL_ITEM_MEAN_FILE,INITIAL_CUSTOMER_OFFSET_FILE,
            ITEM_FEATURE_INITIAL_VALUE, CUSTOMER_FEATURE_INITIAL_VALUE);
   svdModel.loadTrainingData(INITIAL_TRAINING_FILE,DATA_RESIDUAL_INITIAL_VALUE);
   svdModel.run(FEATURE_NUM,EPOCH_NUM,lrate,Kvalue);
   return 0; 
}

