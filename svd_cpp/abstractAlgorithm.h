#ifndef ABSTRACTALGORITHM_H
#define ABSTRACTALGORITHM_H


#include <string>

using namespace std;


class abstractAlgorithm
{
public:
  virtual string currentTime();
  virtual void loadTrainingData(string initialTrainingFile, float initialValue) = 0;
};

#endif
