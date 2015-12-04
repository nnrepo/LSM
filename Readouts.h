#ifndef READOUTS
#define READOUTS

#include "NemoLSM.h"
#include <cblas.h>
/**
 * @class Readout
 * @brief An implementation of a generic readout component.
 * 
 * @details The class is an interface to the basic funcitonality of the 
 * readout. That is, it provides:
 * 	- Training and filtering capabilities
 * 	- Error handling
 * Override the train virtual function to implement learning.
 * 
 * @version 1.0
 * @author Emmanouil Hourdakis
 * @email ehourdak@ics.forth.gr
 */
class Readout {
public:
  LSM 		*_lsm;					///> A pointer to the liquid which the readout is attached
  Training 	*_training;				///> The training data 
  VectorXd 	 _solution;				///> The solution vector
  unsigned int   _samplingInterval;			///> Sampling interval for the filtering of the readout
  
  /**
  * @function Readout
  * @brief Constructor, used to initialize the variables
  * 
  * @param lsm pointer to the liquid
  * @param training the training data
  * @param samplingInterval the sampling interval for the filtering  
  */     
  Readout(LSM *lsm, Training *training, unsigned int samplingInterval) ;
  /**
  * @function train
  * @brief Virtual abstract function
  * 
  * @param filter the filtered data
  * @param trainingVector the training data    
  */         
  virtual VectorXd train(Eigen::MatrixXd filter, VectorXd trainingVector) = 0;
  /**
  * @function trainData
  * @brief train the readout using the training data and the train function 
  * @param layerIndex the index of the layer to use for training
  */           
  MatrixXd trainData(unsigned int layerIndex);  
  /**
  * @function getError
  * @brief returns the error of the readout 
  * @param layerIndex the index of the layer to use for training
  */             
  virtual double getError(MatrixXd solved);
};
/**
 * @class JacobiReadout
 * @brief A Readout descendant, which implements Jacobi 
 * 
 * @version 1.0
 */
class JacobiReadout : public Readout {
public:
  JacobiReadout(LSM *lsm, Training *training, unsigned int samplingInterval);
  virtual VectorXd train(Eigen::MatrixXd filter, VectorXd trainingVector);
};
/**
 * @class CholeskyReadout
 * @brief A Readout descendant, which implements Cholesky factorization 
 * 
 * @version 1.0
 */
class CholeskyReadout : public Readout {
public:
  CholeskyReadout(LSM *lsm, Training *training, unsigned int samplingInterval);
  virtual VectorXd train(Eigen::MatrixXd filter, VectorXd trainingVector);
};
/**
 * @class NormalEquationsReadout
 * @brief A Readout descendant, which implements normal equations 
 * 
 * @version 1.0
 */
class NormalEquationsReadout : public Readout {
public:
  NormalEquationsReadout(LSM *lsm, Training *training, unsigned int samplingInterval);  
  virtual VectorXd train(Eigen::MatrixXd filter, VectorXd trainingVector) ;
};
/**
 * @class BLASLeastSquares
 * @brief A Readout descendant, which implements least squares solving 
 * 
 * @version 1.0
 */
class BLASLeastSquares : public Readout {
public:
  BLASLeastSquares(LSM *lsm, Training *training, unsigned int samplingInterval);  
  virtual VectorXd train(Eigen::MatrixXd filter, VectorXd trainingVector) ;
};

#endif