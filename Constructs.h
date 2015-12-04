#ifndef CONSTRUCTS
#define CONSTRUCTS

#include "Includes.h"
#include "NemoLSM.h"

//Predefined codes for drawing
#define RESET   	"\033[0m"
#define BLACK   	"\033[30m"      	/* Black */
#define RED     	"\033[31m"      	/* Red */
#define GREEN  		"\033[32m"      	/* Green */
#define YELLOW  	"\033[33m"      	/* Yellow */
#define BLUE    	"\033[34m"      	/* Blue */
#define MAGENTA 	"\033[35m"      	/* Magenta */
#define CYAN    	"\033[36m"      	/* Cyan */
#define WHITE   	"\033[37m"      	/* White */
#define BOLDBLACK   	"\033[1m\033[30m"      	/* Bold Black */
#define BOLDRED     	"\033[1m\033[31m"      	/* Bold Red */
#define BOLDGREEN   	"\033[1m\033[32m"      	/* Bold Green */
#define BOLDYELLOW  	"\033[1m\033[33m"      	/* Bold Yellow */
#define BOLDBLUE    	"\033[1m\033[34m"      	/* Bold Blue */
#define BOLDMAGENTA 	"\033[1m\033[35m"      	/* Bold Magenta */
#define BOLDCYAN    	"\033[1m\033[36m"      	/* Bold Cyan */
#define BOLDWHITE   	"\033[1m\033[37m"      	/* Bold White */

//Utility functions
unsigned int 	randRange	(unsigned int min, 	unsigned int max);
int 		randRange	(int min, 		int max);
double 		randRange	(double min, 		double max);
double 		generateGaussian(double mean, 		double deviation);

std::string string_format(const std::string fmt_str, ...);


class LSM;

/**
 * @class GeneralLayer
 * @brief A class to group liquid neurons into a common layer, with topological coordinates.
 * 
 * @details The class implements a layer component for the liquid class.
 * It implements functionalities of an LSM layer:
 * 	- Topological coordinates for each layer, which affect the neurons' positions.
 * 	- Handle Nemo I/O
 * 	- Extracting and recording and stacking in an Eigen sense the neuron values.
 * 
 * @version 1.0
 * @author Emmanouil Hourdakis
 * @email ehourdak@ics.forth.gr
 */
class GeneralLayer {
public:
  int _x;							///> x coordinate of the layer
  int _y;							///> y coordinate of the layer
  int _z;							///> z coordinate of the layer
  
  unsigned int _w;						///> width of the layer
  unsigned int _h;						///> height of the layer
  unsigned int _d;						///> depth of the layer
  
  nemo::Simulation::firing_stimulus  _forced_to_fire;		///> Neurons that will be forced to fire in the layer
  nemo::Simulation::current_stimulus _stimulus_input;		///> Neurons that will fire on demand, based on stimulus
  
  unsigned int  _nNeurons;					///> Number of neurons in the layer
  unsigned int  _minIndex;					///> The index of the first neuron in the layer
  unsigned int  _maxIndex;  					///> The index of the last neuron in the layer
public:
  /**
  * @function Constructor
  * @brief Initializes layer parameters
  * 
  * @param minIndex the index of the first neuron in the liquid
  * @param x the x location of the layer
  * @param y the y location of the layer
  * @param z the z location of the layer
  * @param w the width of the layer
  * @param h the height of the layer
  * @param d the depth of the layer    
  */         
  GeneralLayer(unsigned int minIndex, int x, int y, int z, unsigned int w, unsigned int h, unsigned int d);
  /**
  * @function clear
  * @brief clears input stimulus   
  */       
  void 			clear			();
  /**
  * @function makeVector  
  * @brief stacks neurons' outputs into a vector, assigning
  * 0 and 1 depending on the neuron's firing state  
  * 
  * @param firings the layer neuron firings
  */         
  Eigen::VectorXd 	makeVector		(std::vector<unsigned> firings);  
  /**
  * @function makeMatrix
  *   
  * @brief stacks neurons' outputs into a matrix, assigning
  * 0 and 1 depending on the neuron's firing state. Matrix is
  * ordered by time/row, neurons#/col
  * 
  * @param allFirings the history of layer firings
  */           
  Eigen::MatrixXd 	makeMatrix		(std::vector< std::vector<unsigned> > allFirings);  
  /**
  * @function checkNeuron
  *   
  * @brief checks whether a neuron index belongs to this layer 
  * @param neuronIndex the neuron index to query
  */    
  bool 			checkNeuron		(unsigned int neuronIndex);
  /**
  * @function getNeuronIndexInLSM
  *   
  * @brief returns a neuron's index in the liquid
  * @param neuronIndex the neuron index to query
  */      
  unsigned int 		getNeuronIndexInLSM	(unsigned int neuronIndex);
  /**
  * @function getNeuronPosition
  *   
  * @brief returns neuron position as a 3D vector
  * @param neuronIndex the neuron index to query
  */      
  Eigen::Vector3i 	getNeuronPosition	(unsigned int neuronIndex);
};

/**
 * @class PoissonLayer
 * @brief Derives the GeneralLayer for including Poisson neurons    
 */
class PoissonLayer : public GeneralLayer {
public:    
  /**
  * @function PoissonLayer
  * @brief Constructor
  * @see GeneralLayer constructor for parameter initialization
  */    
  PoissonLayer(unsigned int minIndex, int x, int y, int z, unsigned int w, unsigned int h, unsigned int d);
  /**
  * @function inputNeuron  
  * @brief Set a neuron's firing rate
  */      
  void inputNeuron(LSM *net, unsigned int neuronIndex, float input);
};

/**
 * @class InputLayer
 * @brief Derives the GeneralLayer for including input neurons    
 */
class InputLayer : public GeneralLayer {
public:   
  /**
  * @function InputLayer
  *   
  * @brief Constructor
  * @see GeneralLayer constructor for parameter initialization
  */     
  InputLayer(unsigned int minIndex, int x, int y, int z, unsigned int w, unsigned int h, unsigned int d);
  /**
  * @function fireNeuron
  * @brief Make a neuron fire
  */    
  void fireNeuron(unsigned int neuronIndex);				 
};

/**
 * @class TicToc
 * @brief Use to record and output timings
 */
class TicToc {
public:
  std::stack<clock_t> tictoc_stack;

  /**
  * @function tic
  * @brief Start timer
  */     
  void tic();
  /**
  * @function toc
  * @brief Stop timer
  */     
  void toc(bool bOut=false);
};

#endif