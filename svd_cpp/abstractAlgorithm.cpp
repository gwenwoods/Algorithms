#include "abstractAlgorithm.h"

using namespace std;

string abstractAlgorithm::currentTime()
{
  struct tm *newtime;
  time_t ltime;

  ltime = time(&ltime);
  newtime = localtime(&ltime);
  string myTime = asctime(newtime);
  myTime.erase(24,myTime.size());
  return myTime;
}

