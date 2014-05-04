#include <iostream>
#include <fstream>

#include "abstractAlgorithm.h"
#include "bp.h"

int INPUT_UNIT_NUM;
int HIDDEN_UNIT_NUM;
int OUTPUT_UNIT_NUM;
int DATA_NUM;
int EPOCH_NUM;
float lrate;

string INITIAL_TRAINING_FILE;
string TARGET_FILE;

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
        if(paraName == "INPUT_UNIT_NUM")  {INPUT_UNIT_NUM = atoi(token); cout << "INPUT_UNIT_NUM = " << INPUT_UNIT_NUM << endl;}
        if(paraName == "HIDDEN_UNIT_NUM")  {HIDDEN_UNIT_NUM = atoi(token); cout << "HIDDEN_UNIT_NUM = " << HIDDEN_UNIT_NUM << endl;}
        if(paraName == "OUTPUT_UNIT_NUM")  {OUTPUT_UNIT_NUM = atoi(token); cout << "OUTPUT_UNIT_NUM = " << OUTPUT_UNIT_NUM << endl;}
        if(paraName == "DATA_NUM")  {DATA_NUM = atoi(token); cout << "DATA_NUM = " << DATA_NUM << endl;}
        if(paraName == "EPOCH_NUM")  {EPOCH_NUM = atoi(token); cout << "EPOCH_NUM = " << EPOCH_NUM << endl;}

        if(paraName == "lrate")  {lrate = atof(token); cout << "lrate = " << lrate << endl;}

        if(paraName == "INITIAL_TRAINING_FILE")  { INITIAL_TRAINING_FILE.insert(0,token);
                 cout << "INITIAL_TRAINING_FILE = " << INITIAL_TRAINING_FILE << endl;}

        if(paraName == "TARGET_FILE")  { TARGET_FILE.insert(0,token);
                 cout << "TARGET_FILE = " << TARGET_FILE << endl;}

      }
    }
   input.close();
   BP bpModel(INPUT_UNIT_NUM,HIDDEN_UNIT_NUM,OUTPUT_UNIT_NUM,DATA_NUM); 
   bpModel.loadTrainingData("XOR_training.txt","XOR_target.txt");

   bpModel.run(EPOCH_NUM,lrate);

   ifstream input2("XOR_test.txt");

   float* recallInput = new float[2];
   for (int test=0; test<4; test++){
       input2 >> recallInput[0] >> recallInput[1];
      float *ans = new float[1];
      ans= bpModel.recall(recallInput);
      cout << recallInput[0] << " " << recallInput[1] << " " << ans[0] << endl; 
   }
   
   return 0; 
}

