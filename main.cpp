#include "Includes.h"
#include "NemoLSM.h"
#include "Trainings.h"
#include "Readouts.h"

extern "C" {
#undef FUNPROTO
#include "gaul.h"
}

#define CHECK_MODE(m) if( viewMode & m )
#define USE_GA					//Enable this define to include the genetic algorithm

//Application output mode
enum outMode { outStats 	= 1ul << 0, 
	       outOutputs	= 1ul << 1,
	       outReadout       = 1ul << 2
};
unsigned long 	viewMode 	= outStats;
unsigned long 	lsmMode  	= outStats;
unsigned int 	nGenerations 	= 40;
unsigned int 	nChromosomes 	= 40;
unsigned int 	currentGen   	= 0;
unsigned int 	simTime 	= 1500;    
unsigned int 	sampling 	= 5;
bool 		bSave 		= false;
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

/**
* @function evolve_lsm
* @brief Evolves an LSM, and optimizes it to perform a given function
* 
* @param population pop The population of chromosomes
* @param entity entity The GA pointer
*/ 
boolean evolve_lsm(population *pop, entity *entity) {      
    //Generate training data    
    Training trainings(1, 1);  
//     generateBinaryTraining(trainings, simTime, sampling);
//     generateSinTraining(trainings, simTime, sampling);
    generateMassTraining(trainings, simTime, sampling);
  
    if( currentGen != pop->generation ) {
      currentGen = pop->generation;
    }
    
    printf("\nGeneration %d", pop->generation);
    double *alleles = ((double *)entity->chromosome[0]);
    
    LSM lsm;
    lsm._outMode = lsmMode;
    lsm.addPoissonLayer(0, 0, 0, trainings._nInputs, 1, 1, 
			25.0 + 5.0 * alleles[0]);    	//excitatory percent

    lsm.addLayer(alleles[1], alleles[2], alleles[3], 
		 10, 	     5, 	 5,
		80.0 + 2.0 * alleles[4],
		alleles[4] / 100.0,
		alleles[5] / 20.0,
		alleles[6] * (-8.0),
		alleles[13] / 2.0,
		alleles[14],
		alleles[15] * (-3.0), 0.0 ); 
        
    lsm.connectPoissonAll	(100.0, 		//percent of connections
				 0, 0, 
				alleles[7]/10.0,	//weight mean
				30.0 );       		//weight variance
    lsm.connectPoissonAll	(100.0, 		//percent of connections
				 0, 0, 
				alleles[7]/10.0,	//weight mean
				30.0 );       		//weight variance
    
    lsm.connectLayer		(0, 0,
				1000.0f * alleles[8], 	//nConnections	
				1.0 * alleles[9],	//lambda
				1.0,//alleles[10],   	//distance metric
				alleles[11]/10.0,	//weight mean
				alleles[12]*10.0 );	//weight variance
    
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
    }  
           
    timer.toc();                      

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
    
    printf( RED "\nNormal error: %f\n" RESET, fit );    
    
    entity->fitness = 1.0/fit;
    return TRUE;
}

int main (int argc, char *argv[])
{           
  random_seed(20092004);			/* Random seed requires any integer parameter. 	*/
  
  size_t      *  beststrlen;
  population *pop=NULL;				/* The population of solutions. 		*/

  pop = ga_genesis_double(
    nChromosomes,                     		/* const int              population_size 	*/
    1,                      	 		/* const int              num_chromo 		*/
    16,     					/* const int              len_chromo 		*/
    NULL,                    			/* GAgeneration_hook      			*/
    NULL,                     			/* GAiteration_hook       		 	*/
    NULL,                     			/* GAdata_destructor      		 	*/
    NULL,                 			/* GAdata_ref_incrementor 		 	*/
    evolve_lsm,         			/* GAevaluate             evaluate 		*/
    ga_seed_double_random, 			/* GAseed                 seed 			*/
    NULL,                     			/* GAadapt                	 		*/
    ga_select_one_roulette,     		/* GAselect_one           select_one 		*/
    ga_select_two_roulette,        		/* GAselect_two           select_two 		*/
    ga_mutate_double_singlepoint_drift, 	/* GAmutate  		  mutate 		*/
    ga_crossover_double_mixing, 		/* GAcrossover     	  crossover 		*/
    NULL,                     			/* GAreplace              	 		*/
    NULL                    			/* void *                 	 		*/ );

    ga_population_set_parameters(
    pop,                     			/* population              *pop 		*/
    GA_SCHEME_DARWIN,        			/* const ga_class_type     class 		*/
    GA_ELITISM_PARENTS_DIE,  			/* const ga_elitism_type   elitism 		*/
    0.9,                     			/* double                  crossover 		*/
    0.2,                     			/* double                  mutation 		*/
    0.0               				/* double                  migration 		*/);  
  
  ga_population_set_allele_min_double(pop, 1);
  ga_population_set_allele_max_double(pop, 10);    
  
  ga_evolution(
    pop,                     			/* population              *pop 		*/
    nGenerations                     		/* const int               max_generations 	*/);

  cout << endl << endl << "====================================";
  printf( "The final solution found was:\n");
  printf( "Fitness score = %f Error: %f\n", 
	  ga_get_entity_from_rank(pop,0)->fitness,
	  1.0 / ga_get_entity_from_rank(pop,0)->fitness );

  entity *best = ga_get_entity_from_rank(pop,0);
  double *best_alleles = ((double *)best->chromosome[0]);
  
  cout << endl << "Run best solution?" << endl;
  cin.get();
  
  char best_chromosome[256];
  size_t str_length;
  ga_chromosome_double_to_string(pop, ga_get_entity_from_rank(pop,0), best_chromosome, &str_length);
  cout << "Best chromosome is: " << best_chromosome << endl;
  cin.get();

  bSave 	= true;
  lsmMode 	= LSM::outNone;
  viewMode 	= outStats | outOutputs;
  evolve_lsm(pop, best); 
  cin.get();
  
  ga_extinction(pop);	/* Deallocates all memory associated with the population and it's entities. */       

  return 0;
}
