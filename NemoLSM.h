#ifndef NEMO_LSM
#define NEMO_LSM

//General program includes
#include "Includes.h"

//LSM stuff
#include "Constructs.h"
#include "Trainings.h"

//Declarations to resolve circular references
class NemoSim;
class GeneralLayer;
class InputLayer;
class PoissonLayer;

/**
 * @class LSM
 * @brief An implementation of a Liquid State Machine, using NeMo. 
 * 
 * @details The class implements the liquid component of a Liquid State Machine.
 * It implements several functionalities of a liquid:
 * 	- Neurons are assigned topological positions and distributed on a 3D grid.
 * 	- Synapse initialization is affected by the pre- and post-synaptic neuron properties
 * 	- Storing and handling the neuron outputs
 * 	- Sampling and filtering functions
 * 	- Arbitrary connectivity between different parts of the liquid
 * 	- 
 * 
 * @version 1.0
 * @author Emmanouil Hourdakis
 * @email ehourdak@ics.forth.gr
 */
class LSM {
  friend NemoSim;							///> Give access to NemoSim class
  friend GeneralLayer;							///> Give access to GeneralLayer 
  friend PoissonLayer;							///> Give access to PoissonLayer  
  
public:  
  enum netOutMode { outNone = 1ul << 0,					///> Output to (i) nothing, (ii) std::cout, (iii) file
		    outCout = 1ul << 1, 
		    outFile = 1ul << 2 };  
protected:
  unsigned int 					_currentNeuronIndex;	///> Internal index, used when initializing the neurons
  
  unsigned 					_mPoisson_type;		///> Nemo's ID for the Poisson type of neuron
  unsigned 					_mInput_type;		///> Nemo's ID for the Input type of neuron
  unsigned 					_mIF_type;		///> Nemo's ID for the Integrate and Fire type of neuron  
  
  nemo::Network  				_net;			///> The nemo network object
  nemo::Configuration 				_conf;			///> The nemo network configuration
  NemoSim 				       *_sim;			///> The nemo simulation object  
  
  std::ofstream 				_outfile;		///> If the mode supports it, an output file to write the liquid's output  
  unsigned int   				_nNeurons;		///> The number of neurons held by the liquid
  unsigned int   				_nConnections;		///> The number of connections maintained by the liquid
  unsigned int 					_inputNeurons;		///> The number of input neurons to the liquid
  
public:
  unsigned long					_outMode;		///> The mode used by the liquid to printout information @see LSM::netOutMode
      
  Eigen::MatrixXd				_spikeFirings; 		///> Matrix holding neurons fired as ones ... etc  
  vector<bool>					_excitatoryConnections;	///> Vector of booleans, excitatory (true) - inhibitory (false), for each liquid connection
  
  unsigned int 					_simTime;		///> The time that the liquid has been simulated
  
  vector<GeneralLayer> 				_layers;		///> The liquid's layers
  vector<InputLayer> 				_inputLayers;		///> The liquid's input layers
  vector<PoissonLayer>				_poissonLayers;		///> The liquid's poisson layers    
  
  std::vector<unsigned> 			_fired;			///> A vector of neuron firings on each simulation step
  std::vector< std::vector<unsigned> > 		_firings;		///> A history of neuron firings (firings of firings) for the liquid
  
  //Liquid cosntructor/destructor
  LSM						( ); 
  ~LSM						( );  
    
  /**
  * @function out
  * @brief Function used to output information of the liquid. 
  * @details The function will output in different streams, depending
  * on the netOutMode parameter.
  * @param outInfo the string to be output.
  * @see LSM::netOutMode
  */       
  void out					(std::string outInfo);
  
  /**
  * @function addLayer
  * @brief Adds a layer with Izhikevich neurons in the liquid.
  * @details The function assigns topological coordinates to neurons
  * within the liquid and distributes them based on the w, h and d parameters.
  * It also configures the type of synapses (excitatory/inhibitory) based on 
  * the excitatory percent passed. 
  * 
  * @param x the x coordinate of the layer's positions
  * @param y the y coordinate of the layer's positions
  * @param z the z coordinate of the layer's positions
  * 
  * @param w the width of the layer
  * @param h the height of the layer
  * @param d the depth of the layer
  * 
  * @param excitatoryPercent percent of excitatory connections in the liquid
  * 
  * @param a 		Nemo's Izhikevich neuron property
  * @param b 		Nemo's Izhikevich neuron property
  * @param c 		Nemo's Izhikevich neuron property
  * @param dd		Nemo's Izhikevich neuron property
  * @param u		Nemo's Izhikevich neuron property
  * @param v		Nemo's Izhikevich neuron property
  * @param sigma	Currently not used
  * 
  * @see Nemo::Izhikevich class for more details on the neuron type
  */       
  void addLayer                    		( int x, int y, int z, unsigned int w, unsigned int h, unsigned int d, double excitatoryPercent,
						  float a, float b, float c, float dd, float u, float v, float sigma);

  /**
  * @function addLayerGaussian
  * @brief Adds a layer with Izhikevich neurons in the liquid, 
  * drawing the initialization parameters of the neurons from 
  * a Gaussian distribution dedicated to each parameter.
  * @details The function performs similarly to addLayer, but
  * initializes each parameter from values that are drawn from a 
  * Gaussian distribution, to increase the variability within the 
  * neurons.
  * 
  * @see For parameters x, y, z, w, h, d, excitatoryPercent see addLayer funciton
  * 
  * @param meanDeviation 				The mean deviation for every distribution
  * @param timeScaleofRecovery 				The mean of the time scale of recovery distribution
  * @param sensitivityToSubthresholdFlucutations 	The mean of the sensitivity to fluctuations distribution
  * @param afterSpikeResetValueOfMembranePotential	The mean of the after spike membrane potential reset value distribution
  * @param afterSpikeResetOfRecovery			The mean of the after spike recovery distribution
  * @param initialValueOfMembraneRecovery		The mean of the initial membrane value distribution
  * @param sigmaForRandomGaussianPerNeuronInput		The mean of the random per neuron input distribution
  * 
  * @see Nemo::Izhikevich class for more details on the neuron type
  */         
  void addLayerGaussian 			( int x, int y, int z, unsigned int w, unsigned int h, unsigned int d,
						  double excitatoryPercent, 
						 double meanDeviation,				double timeScaleofRecovery, 			
						 double sensitivityToSubthresholdFlucutations,	double afterSpikeResetValueOfMembranePotential, 	
						 double afterSpikeResetOfRecovery,		double initialValueOfMembraneRecovery, 		
						 double initialValueOfMembranePotential,	double sigmaForRandomGaussianPerNeuronInput);
  
  /**
  * @function addLayerIF
  * @brief Adds a layer with Integrate and Fire neurons in the liquid.
  * @details The function initializes the IF neurons using a distribution
  * that has a mean for each parameter and some large standard deviation. Custom 
  * values for the mean of the distribution of some parameters can be specified by
  * the user (see function parameters below).
  * 
  * @see For parameters x, y, z, w, h, d, excitatoryPercent see addLayer funciton
  * 
  * @param v_thresh 				Nemo's IF neuron property
  * @param v_reset 				Nemo's IF neuron property
  * @param refraq 				Nemo's IF neuron property
  * @param tau_m				Nemo's IF neuron property
  * @param inhibitoryCurrent			Nemo's IF neuron property
  * 
  * @see Nemo::Izhikevich class for more details on the neuron type
  * @todo Need to fix the distribution initialization for all parameter types.
  */    
  void addLayerIF				( int x, int y, int z, unsigned int w, unsigned int h, unsigned int d, 
						  double excitatoryPercent, double v_thresh, double v_reset, double refraq, 
						  double tau_m, double inhibitoryCurrent );
  
  /**
  * @function addInputLayer
  * @brief Adds a layer of input neurons to the liquid
  * @see For parameters x, y, z, w, h, d, excitatoryPercent see addLayer funciton
  * 
  * @todo Need to fix the distribution initialization for all parameter types.
  */    
  void addInputLayer				( int x, int y, int z, unsigned int w, unsigned int h, unsigned int d,
						  double excitatoryPercent  );  
  /**
  * @function addPoissonLayer
  * @brief Adds a layer of Poisson input neurons to the liquid
  * @see For parameters x, y, z, w, h, d, excitatoryPercent see addLayer funciton
  * 
  * @todo Need to fix the distribution initialization for all parameter types.
  */      
  void addPoissonLayer 				( int x, int y, int z, unsigned int w, unsigned int h, unsigned int d,
						  double excitatoryPercent  ); 
  /**
  * @function addPoissonLayer
  * @brief Generic connection function, connects two layers based 
  * on the specifications of the parameters. The other functions
  * use this one.
  *
  * @param fromLayerS 		Index of the presynaptic layer
  * @param toLayerS 		Index of the post-synaptic layer
  * @param nConnections 	Number of connections to form
  * @param lambda		How much distance affects the propability of forming a connection
  * @param distanceMetric	How much distance affects the synapse's delay
  * @param weightMean		Mean of the distribution used to draw the synapse weight initial value
  * @param weightVariance	Variance of the distribution used to draw the synapse weight initial value
  * @param bLearning		Enable learning in the connection
  */       
  void connectLayerGeneric			( GeneralLayer &fromLayerS,     GeneralLayer &toLayerS,
						  unsigned int nConnections,  	
						  unsigned int lambda, 	      	double distanceMetric,
						  double weightMean, 	      	double weightVariance,
						  bool bLearning = false ); 
  /**
  * @function connectLayer
  * @brief Similar to the connectLayerGeneric function, but using indices 
  * instead of layer pointers.
  * @see connectLayerGeneric for the remaining parameters
  */      
  void connectLayer				( unsigned int fromLayer,     unsigned int toLayer, 
						  unsigned int nConnections,  	
						  unsigned int lambda, 	      double distanceMetric,
						  double weightMean, 	      double weightVariance,
						  bool bLearning = false);  
  /**
  * @function interConnectLayer
  * @brief Similar to the connectLayerGeneric function, but used for interconnecting
  * the neurons within the same layer.
  * @see connectLayerGeneric for the remaining parameters
  */        
  void interConnectLayer			( unsigned int layer, 
						  unsigned int nConnections,  	
						  unsigned int lambda, 	      double distanceMetric,
						  double weightMean, 	      double weightVariance,
						  bool bLearning = false);     
  /**
  * @function connectInput
  * @brief Similar to the connectLayerGeneric function, but used for connecting input and 
  * liquid layers.
  * @see connectLayerGeneric for the remaining parameters
  */          
  void connectInput 				( unsigned int fromLayer,     unsigned int toLayer, 
						  unsigned int nConnections,  	
						  unsigned int lambda, 	      double distanceMetric,
						  double weightMean, 	      double weightVariance,
						  bool bLearning = false );  
  /**
  * @function connectPoisson
  * @brief Similar to the connectLayerGeneric function, but used for connecting poisson input and 
  * liquid layers.
  * @see connectLayerGeneric for the remaining parameters
  */      
  void connectPoisson 				( unsigned int fromLayer,  	unsigned int toLayer, 
						  unsigned int nConnections,  	
						  unsigned int lambda, 	  	double distanceMetric,
						  double weightMean, 	      	double weightVariance,
						  bool bLearning = false );  
  /**
  * @function connectPoissonAll
  * @brief Similar to the connectLayerGeneric function, but used for connecting poisson input and 
  * liquid layers.
  * @see connectLayerGeneric for the remaining parameters
  */        
  void connectPoissonAll			( double connectionPercentage,
						  unsigned int fromLayer,  	unsigned int toLayer, 
						  double weightMean, 	        double weightVariance,
						  bool bLearning  = false );       
    
  /**
  * @function init
  * @brief Initializes the liquid network on the Nemo library.
  * @details This function must be called after the creation of an LSM layer.  
  */          
  void init 					( NemoSim *sim);	
  /**
  * @function priorToSim
  * @brief Clears all network stored inputs, prior to a new simulation run.    
  */          
  void priorToSim				( );		  
  /**
  * @function simulate
  * @brief Simulates one cycle of the liquid network
  * @details The function iterates through all input and poisson layers
  * and setsup the neurons that need to fire within the liquid. 
  */         
  const std::vector<unsigned> simulate		( unsigned int nsteps = 1 );  
  /**
  * @function filter
  * @brief Filters all liquid output based on the filterFun.  
  */       
  Eigen::MatrixXd filter			( unsigned int );
  /**
  * @function filterFun
  * @brief Filter function used to filter the neuron outputs in the liquid,
  * used by the filter function.
  * 
  * @see filter function for details on how filtering is accomplished.
  */     
  double filterFun				( double inValue );
};

/**
 * @class NemoSim
 * @brief A wrapper for class for the nemo::Simulation object
 * 
 * @version 1.0
 */
class NemoSim {
public:
  LSM *_lsm;
  boost::scoped_ptr <nemo::Simulation>  	_sim;
  
  /**
  * @function NemoSim
  * @brief Constructor of the class, used to set the LSM pointer
  */      
  NemoSim(LSM *lsm)
  : _lsm(lsm),
    _sim ( nemo :: simulation ( lsm->_net , lsm->_conf )) {    
  }
  
  /**
  * @function Destructor
  * @brief Destructor of the class (currently does nothing)
  */     
  ~NemoSim() {
  }
};

#endif