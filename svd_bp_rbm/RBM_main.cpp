#include <iostream>
#include <fstream>

#include "abstractAlgorithm.h"
#include "rbm.h"

int VISIBLE_UNIT_NUM;
int HIDDEN_UNIT_NUM;
int RECORD_NUM;
int EPOCH_NUM;
float lrate;
float DATA_RESIDUAL_INITIAL_VALUE;

string INITIAL_TRAINING_FILE;
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
        if(paraName == "VISIBLE_UNIT_NUM")  {VISIBLE_UNIT_NUM = atoi(token); cout << "VISIBLE_UNIT_NUM = " << VISIBLE_UNIT_NUM << endl;}
        if(paraName == "HIDDEN_UNIT_NUM")  {HIDDEN_UNIT_NUM = atoi(token); cout << "HIDDEN_UNIT_NUM = " << HIDDEN_UNIT_NUM << endl;}
        if(paraName == "RECORD_NUM")  {RECORD_NUM = atoi(token); cout << "RECORD_NUM = " << RECORD_NUM << endl;}
        if(paraName == "EPOCH_NUM")  {EPOCH_NUM = atoi(token); cout << "EPOCH_NUM = " << EPOCH_NUM << endl;}

        if(paraName == "lrate")  {lrate = atof(token); cout << "lrate = " << lrate << endl;}

        if(paraName == "INITIAL_TRAINING_FILE")  { INITIAL_TRAINING_FILE.insert(0,token);
                 cout << "INITIAL_TRAINING_FILE = " << INITIAL_TRAINING_FILE << endl;}

        if(paraName == "DATA_RESIDUAL_INITIAL_VALUE")  {DATA_RESIDUAL_INITIAL_VALUE = atof(token);
                 cout << "DATA_RESIDUAL_INITIAL_VALUE = " << DATA_RESIDUAL_INITIAL_VALUE << endl;}
      }
    }
   input.close();
   RBM rbmModel(VISIBLE_UNIT_NUM,HIDDEN_UNIT_NUM,RECORD_NUM); 
   rbmModel.loadTrainingData(INITIAL_TRAINING_FILE);
   rbmModel.run(EPOCH_NUM,lrate);
   float **recordStateV_neg;
   recordStateV_neg = new float*[RECORD_NUM];
   for(int i = ZERO; i < RECORD_NUM; i++) {
      recordStateV_neg[i] = new float[VISIBLE_UNIT_NUM];
   }

   recordStateV_neg = rbmModel.getRecordStateV_neg();
   int count110 = 0;
   int count101 = 0;
   int count011 = 0;
   int count000 = 0;

   int count111 = 0;
   int count100 = 0;
   int count010 = 0;
   int count001 = 0;

   for(int i = 0; i < 1000 ; i++ ) {
      if( recordStateV_neg[i][0] == 1  && recordStateV_neg[i][1] == 1  && recordStateV_neg[i][2] == 0) { count110++;}
      if( recordStateV_neg[i][0] == 1  && recordStateV_neg[i][1] == 0  && recordStateV_neg[i][2] == 1) { count101++;}
      if( recordStateV_neg[i][0] == 0  && recordStateV_neg[i][1] == 1  && recordStateV_neg[i][2] == 1) { count011++;}
      if( recordStateV_neg[i][0] == 0  && recordStateV_neg[i][1] == 0  && recordStateV_neg[i][2] == 0) { count000++;}

      if( recordStateV_neg[i][0] == 1  && recordStateV_neg[i][1] == 1  && recordStateV_neg[i][2] == 1) { count111++;}
      if( recordStateV_neg[i][0] == 1  && recordStateV_neg[i][1] == 0  && recordStateV_neg[i][2] == 0) { count100++;}
                if( recordStateV_neg[i][0] == 0  && recordStateV_neg[i][1] == 1  && recordStateV_neg[i][2] == 0) { count010++;}
                if( recordStateV_neg[i][0] == 0  && recordStateV_neg[i][1] == 0  && recordStateV_neg[i][2] == 1) { count001++;}

            }

            cout << "110  " << count110 << endl;
            cout << "101  " << count101 << endl;
            cout << "011  " << count011 << endl;
            cout << "000  " << count000 << endl;

            cout << "111  " << count111 << endl;
            cout << "100  " << count100 << endl;
            cout << "010  " << count010 << endl;
            cout << "001  " << count001 << endl;



   
   return 0; 
}

