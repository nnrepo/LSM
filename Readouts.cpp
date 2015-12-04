#include "Readouts.h"

Readout::Readout(LSM *lsm, Training *training, unsigned int samplingInterval) 
: _lsm(lsm),
  _training(training),
  _samplingInterval(samplingInterval) {
}  

MatrixXd Readout::trainData(unsigned int layerIndex) {
//Filter the data
  Eigen::MatrixXd beforeFilter 	= _lsm->_layers[layerIndex].makeMatrix(_lsm->_firings);
  Eigen::MatrixXd filter 	= _lsm->filter(layerIndex);  
  Eigen::MatrixXd sampled(filter.rows(), filter.cols()/_samplingInterval);
  for( unsigned i = 0; i < filter.cols()/_samplingInterval; i++ ) {
      sampled.col(i) = filter.col(i*_samplingInterval);
  }
#ifdef DEBUG__4    
  cout << endl << "Before filtering" << endl << beforeFilter;
  cout << endl << "After filtering"  << endl << filter << endl;
#endif
  
  VectorXd trainingVector = _training->generateOutputVector(_samplingInterval);
#ifdef DEBUG__4
  cout << endl << "Training vector" << endl << trainingVector;
#endif    
  
  _solution = train(sampled, trainingVector);
  
  MatrixXd solved = sampled.transpose() * _solution;
  return solved;
}

double Readout::getError(MatrixXd solved) {
  VectorXd trainingVector = _training->generateOutputVector(_samplingInterval);
  MatrixXd checkResult = solved - trainingVector;
#ifdef DEBUG__4
  cout << endl << "After evaluation: " << endl << checkResult << endl;
#endif    
  double fit = checkResult.squaredNorm()/checkResult.rows();
  return fit;
}

JacobiReadout::JacobiReadout(LSM *lsm, Training *training, unsigned int samplingInterval)
: Readout(lsm, training, samplingInterval) {
}

VectorXd JacobiReadout::train(Eigen::MatrixXd filter, VectorXd trainingVector) {
#ifdef DEBUG__1  
  cout << "Training least squares with svd" << endl;
#endif
  VectorXd solution = filter.jacobiSvd(ComputeThinU | ComputeThinV).solve(trainingVector);
  
  return solution;
}

CholeskyReadout::CholeskyReadout(LSM *lsm, Training *training, unsigned int samplingInterval)
: Readout(lsm, training, samplingInterval) {
}

VectorXd CholeskyReadout::train(Eigen::MatrixXd filter, VectorXd trainingVector) {
#ifdef DEBUG__1
  cout << "Training least squares with choleski" << endl;
#endif
  VectorXd solution = filter.transpose().colPivHouseholderQr().solve(trainingVector); 
  
  return solution;
}

NormalEquationsReadout::NormalEquationsReadout(LSM *lsm, Training *training, unsigned int samplingInterval)
: Readout(lsm, training, samplingInterval) {
}

VectorXd NormalEquationsReadout::train(Eigen::MatrixXd filter, VectorXd trainingVector) {
#ifdef DEBUG__1  
  cout << "Training least squares with normal equations" << endl;
#endif
  VectorXd solution = (filter.transpose() * filter).ldlt().solve(filter.transpose() * trainingVector);  
  
  return solution;
}

BLASLeastSquares::BLASLeastSquares(LSM *lsm, Training *training, unsigned int samplingInterval)
: Readout(lsm, training, samplingInterval) {
}

VectorXd BLASLeastSquares::train(Eigen::MatrixXd filter, VectorXd trainingVector) {
#ifdef DEBUG__1  
  cout << "Training least squares with normal equations" << endl;
#endif
  
  MatrixXd mat(3,3);
  mat << 3, 1, 3, 1, 5, 9, 2, 6, 5;
  VectorXd vec(3);
  vec << -1, -1, 1;
  cout << mat;
  cout << endl << vec << endl;
  
  VectorXd solution = (mat.transpose() * mat).ldlt().solve(mat.transpose() * vec);  
  cout << solution << endl;
  
  VectorXd solution2 = mat.colPivHouseholderQr().solve(vec);   
  cout << solution2 << endl;
  cout << mat * solution2 << endl;
// double data[12];
// MatrixXd solution2 = Map<Matrix<double,4,3,RowMajor>>(m);
// Map<MatrixXd>( data, solution2.cols(), solution2.rows() ) =   solution2.transpose();
// cout << solution2;

//   MatrixXd solution = Map<Matrix<double, 4, 1>>(y);
//   cout << solution;
  return solution.col(0);
}

