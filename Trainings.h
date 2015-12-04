#ifndef TRAININGS
#define TRAININGS

#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
using namespace Eigen;
#include <vector>
using namespace std;

class Training;

/**
 * @class IOPair
 * @brief Class used to pair input/output data for training.
 * 
 * @version 1.0 
 */
class IOPair {
public:
  Eigen::VectorXd _inputs;
  Eigen::VectorXd _outputs;
  
  IOPair(Training *training); 
  IOPair(unsigned int nInputs, unsigned int nOutputs);  
  IOPair(Training *training, double inputs[], double outputs[]) ;
};
/**
 * @class Training
 * @brief Class used to form a complete training dataset.
 * 
 * @version 1.0 
 */
class Training {
public:
  unsigned int 	 _nInputs;
  unsigned int 	 _nOutputs;
  
  vector<IOPair> _trainings;
  
  Training(unsigned int nInputs, unsigned int nOutputs);
  void addTraining(IOPair training);    
  VectorXd generateOutputVector(unsigned int samplingInterval);
};


#endif