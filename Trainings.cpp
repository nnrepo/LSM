#include "Trainings.h"

IOPair::IOPair(Training *training) 
: _inputs (VectorXd::Zero(training->_nInputs)),
  _outputs(VectorXd::Zero(training->_nOutputs)) {  
}

IOPair::IOPair(unsigned int nInputs, unsigned int nOutputs) 
: _inputs (VectorXd::Zero(nInputs)),
  _outputs(VectorXd::Zero(nOutputs)) {    
}

IOPair::IOPair(Training *training, double inputs[], double outputs[]) 
: _inputs (VectorXd::Zero(training->_nInputs)),
  _outputs(VectorXd::Zero(training->_nOutputs)) { 
}

Training::Training(unsigned int nInputs, unsigned int nOutputs) 
: _nInputs (nInputs),
  _nOutputs(nOutputs) {
}
    
void Training::addTraining(IOPair training) {
  _trainings.push_back(training);
}

VectorXd Training::generateOutputVector(unsigned int samplingInterval) {
  VectorXd ret(_trainings.size()/samplingInterval);
  
  for( unsigned i = 0; i < _trainings.size()/samplingInterval; i++ ) {	
    ret(i) = _trainings[i*samplingInterval]._outputs(0);      
  }
  return ret;
}