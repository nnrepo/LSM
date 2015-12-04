#include "Includes.h"
#include "NemoLSM.h"
#include "Trainings.h"
#include "Readouts.h"

#define CHECK_MODE(m) if( viewMode & m )

//Application output mode
enum outMode { outStats 	= 1ul << 0, 
	       outOutputs	= 1ul << 1, 
	       outDrawing	= 1ul << 2,
	       outReadout       = 1ul << 3
};
unsigned long 	viewMode 	= outStats | outOutputs;
unsigned long 	lsmMode  	= outStats;
unsigned int 	nGenerations 	= 40;
unsigned int 	nChromosomes 	= 40;
unsigned int 	currentGen   	= 0;
unsigned int 	simTime 	= 1500;    
unsigned int 	sampling 	= 5;
bool 		bSave 		= true;
vector<double> 	all_values;
TicToc 	    	timer;

/**
* @function generateBinaryTraining
* @brief Generates training data for a learning a classification task
* 
* @param trainings pointer to the training data to be filled
* @param simTime time for which the synthetic data will be produced
* @param sampleing sampling interval for the data
*/     
void generateBinaryTraining(Training &trainings, unsigned int simTime, unsigned int sampling) {
    for( unsigned i = 0; i < simTime; i++ ) {
      IOPair training(&trainings);  
      training._inputs(0)  = (i<(simTime/2.0)) ? 0.02 : 0.6;  
      training._outputs(0) = (i<(simTime/2.0)) ?  1.0 : 0.0;
      trainings.addTraining(training);
    }
}
/**
* @function generateSinTraining
* @brief Generates training data for a learning a sinusoidal function
* 
* @param trainings pointer to the training data to be filled
* @param simTime time for which the synthetic data will be produced
* @param sampleing sampling interval for the data
*/     
void generateSinTraining(Training &trainings, unsigned int simTime, unsigned int sampling) {
  int backInTime = 200;
  all_values.clear();
  for( unsigned i = 0; i < simTime/sampling; i++ ) {
    double value = randRange(-3.14, 3.14);  
    all_values.push_back(value);
    double backInTimeValue = all_values[(i<backInTime)?0:i-backInTime];
   
    for( unsigned int j = 0; j < sampling; j++ ) {
      IOPair training(&trainings);  	
      training._inputs(0)  = value;
      training._outputs(0) = (i<backInTime)?0.0:sin(backInTimeValue);// + cos(value);//value);
      trainings.addTraining(training);	
    }
  }
}
/**
* @function generateMassTraining
* @brief Generates data as discussed in the Mass paper
* 
* @param trainings pointer to the training data to be filled
* @param simTime time for which the synthetic data will be produced
* @param sampleing sampling interval for the data
*/     
void generateMassTraining(Training &trainings, unsigned int simTime, unsigned int sampling) {
  trainings._nInputs = 8;  
  for( unsigned i = 0; i < simTime/sampling; i++ ) {
    double val1 = randRange(0.0, 1.0);
    double val2 = randRange(0.0, 1.0);
    double val3 = randRange(0.0, 1.0);
    
    for( unsigned j = 0; j < sampling; j++ ) {      
	IOPair training(&trainings);  	
	training._inputs(0)  = val1;
	training._inputs(1)  = val2;
	training._inputs(2)  = val3;
	training._outputs(0) = 10.0*val1+10.0*val2+10.0*val3;
	trainings.addTraining(training);		
      }
    }
}

int main (int argc, char *argv[])
{           
  //Generate training data    
  Training trainings(1, 1);  
//   generateBinaryTraining(trainings, simTime, sampling);
  generateSinTraining(trainings, simTime, sampling);
//   generateMassTraining(trainings, simTime, sampling);
          
  double alleles[16] = {6.131599, 7.394816, 2.184467, 4.553420, 2.660279, 1.068094, 1.204304, 2.400204, 4.776328, 6.516498, 8.549516, 6.460633, 8.381927, 5.145809, 4.437454, 2.312391};
    
  LSM lsm;
  lsm._outMode = lsmMode;
  lsm.addPoissonLayer(0, 0, 0, trainings._nInputs, 1, 1, 
		      25.0 + 5.0 * alleles[0]);
  lsm.addLayer(alleles[1], alleles[2], alleles[3], 
	      10, 5, 5,
	      80.0 + 2.0 * alleles[4],
	      alleles[4] / 100.0,
	      alleles[5] / 20.0,
	      alleles[6] * (-8.0),
	      alleles[13] / 2.0,
	      alleles[14],
	      alleles[15] * (-3.0), 0.0 ); 
      
  lsm.connectPoissonAll	(100.0, 		
			 0, 0, 
			 alleles[7]/10.0,	
			 30.0 );       		
  lsm.connectPoissonAll	(100.0, 		
			 0, 0, 
			 alleles[7]/10.0,	
			 30.0 );       		
  
  lsm.connectLayer	(0, 0,
			 1000.0f * alleles[8], 	
			 1.0 * alleles[9],	
			 1.0,		  	
			 alleles[11]/10.0,	
			 alleles[12]*10.0 );	
  
  NemoSim sim(&lsm);
  lsm.init(&sim);          
  
  //Start simulating and training the network
  timer.tic();
  CHECK_MODE(outStats) cout << endl << "Simulating network" << endl;
  for( unsigned i = 0; i < trainings._trainings.size(); i++ ) {
    lsm.priorToSim();
    for( unsigned n = 0; n < lsm._poissonLayers[0]._nNeurons; n++ ) {
	lsm._poissonLayers[0].inputNeuron(&lsm, n, (trainings._trainings[i]._inputs(n)));
    }        
    
    std::vector<unsigned> fired = lsm.simulate();
    std::sort(fired.begin(), fired.end());
    
#ifdef DEBUG__
    printf("Simtime: %u Neurons fired: %d     --- ", i, (int)fired.size());
    for( unsigned i = 0; i < fired.size(); i++ ) {
      printf("%u ", fired[i]);
    }
    printf("\n");
#endif
  }
           
  timer.toc(true);                      

  CholeskyReadout readout(&lsm, &trainings, sampling); 
  MatrixXd solved	= readout.trainData(0);    
  double fit 		= readout.getError(solved);        
  
  VectorXd trainingVector = trainings.generateOutputVector(readout._samplingInterval);
  
  CHECK_MODE(outOutputs) {
    MatrixXd illus(simTime/readout._samplingInterval, 2);
    illus.col(0) = solved.col(0);
    illus.col(1) = trainingVector;
    cout << illus << endl;
  }
    
  if( bSave ) {
    std::ofstream bestfile;
    bestfile.open("best.m", std::ofstream::out | std::ofstream::trunc);
    MatrixXd illus(simTime/readout._samplingInterval, 2);
    illus.col(0) = solved.col(0);
    illus.col(1) = trainingVector;
    bestfile << "a=[ " << illus << endl << "];" << endl;     
  }    

  printf( RED "\nReadout error: %f\n" RESET, fit );        

  return 0;
}
