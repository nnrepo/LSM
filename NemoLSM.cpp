#include "NemoLSM.h"

#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random.hpp>
#include <ctime>            // std::time


void LSM::out(std::string outInfo) {
  if( _outMode & outCout ) {
    cout << endl << outInfo.c_str();
  }
  if( _outMode & outFile ) {
    _outfile << endl << outInfo.c_str();    
  }
}

void LSM::addLayer( int x, int y, int z, unsigned int w, unsigned int h, unsigned int d, double excitatoryPercent,
  float a, float b, float c, float dd, float u, float v, float sigma ) {
  
  GeneralLayer newLayer(_currentNeuronIndex, x, y, z, w, h, d);
  
  for( unsigned i = 0; i < newLayer._nNeurons; i++ ) {
//     _net.addNeuron (_currentNeuronIndex+i , );
    _net.addNeuron (_currentNeuronIndex+i , a, b, c, dd, u, v, 0.0);
    bool isExcitatory = (randRange(0.0, 100.0)>(100.0-excitatoryPercent));
    _excitatoryConnections.push_back(isExcitatory);  
  }      
  
  _currentNeuronIndex     += newLayer._nNeurons;
  _nNeurons         += newLayer._nNeurons;
  _layers.push_back(newLayer);  
}

void LSM::addLayerGaussian ( int x, int y, int z, unsigned int w, unsigned int h, unsigned int d,
			    double excitatoryPercent,
			    double meanDeviation,				double timeScaleofRecovery, 			
			    double sensitivityToSubthresholdFlucutations,	double afterSpikeResetValueOfMembranePotential, 	
			    double afterSpikeResetOfRecovery,			double initialValueOfMembraneRecovery, 		
			    double initialValueOfMembranePotential,		double sigmaForRandomGaussianPerNeuronInput ) {
  GeneralLayer newLayer(_currentNeuronIndex, x, y, z, w, h, d); 
  
  for( unsigned i = 0; i < newLayer._nNeurons; i++ ) {    
//     0.02 ,0.20 , -61.3 ,6.5 , -13.0 , -65.0 ,0.0);    
    _net.addNeuron (_currentNeuronIndex+i , generateGaussian(timeScaleofRecovery, 			meanDeviation),
					    generateGaussian(sensitivityToSubthresholdFlucutations, 	meanDeviation),
					    generateGaussian(afterSpikeResetValueOfMembranePotential, 	meanDeviation),
					    generateGaussian(afterSpikeResetOfRecovery, 		meanDeviation),
					    generateGaussian(initialValueOfMembraneRecovery, 		meanDeviation),
					    generateGaussian(initialValueOfMembranePotential, 		meanDeviation),
					    generateGaussian(sigmaForRandomGaussianPerNeuronInput,	meanDeviation));
      bool isExcitatory = (randRange(0.0, 100.0)>(100.0-excitatoryPercent));
      _excitatoryConnections.push_back(isExcitatory);		   
  }  
  
  _currentNeuronIndex 	+= newLayer._nNeurons;
  _nNeurons 		+= newLayer._nNeurons;
  _layers.push_back(newLayer);   
}

void LSM::addLayerIF ( int x, int y, int z, unsigned int w, unsigned int h, unsigned int d, 
		       double excitatoryPercent, double v_thresh, double v_reset, double refraq, double tau_m, double inhibitoryCurrent) {
  GeneralLayer newLayer(_currentNeuronIndex, x, y, z, w, h, d); 
    
  for( unsigned i = 0; i < newLayer._nNeurons; i++ ) {
// 	const float args[13] = {
// 		-65.0f, 1.0f, 20.0f, 5.0f, 2.0f, 5.0f, 0.1f, -70.0f, -51.0f,
// 		-65.0f, 0.0f, 0.0f, 1000.0f }
    
    float args[13];
    
    args[0]  = -65.0f;		//  v_rest     : Resting membrane potential in mV.
    args[1]  =  1.0f;		//  cm         : Capacitance of the membrane in nF
    args[2]  =  tau_m; //20.0f;		//  tau_m      : Membrane time constant in ms.
    args[3]  =  refraq;//5.0f;		//  tau_refrac : Length of refractory period in ms.
    args[4]  =  1.0f;		//  tau_syn_E  : Decay time of excitatory synaptic current in ms.
    args[5]  =  1.0f;		//  tau_syn_I  : Decay time of inhibitory synaptic current in ms.
    args[6]  =  0.1f;		//  i_offset   : Offset current in nA
    args[7]  =  v_reset;	//  -70.0f;    : Reset potential after a spike in mV.
    args[8]  =  v_thresh;	//  -21.0f;    : Spike threshold in mV.    
	
    args[9]  = -65.0f;		//  v  : membrane potential
    args[10] =  0.0f;		//  ie : excitatory current
    args[11] =  inhibitoryCurrent;//0.0f;		//  ii : inhibitory current
    args[12] =  1000.0f;	//  lastfire : number of cycles since last firing    
    
    for( unsigned i = 0; i < 13; i++ ) {
      args[i] = generateGaussian(args[i], 5.0);
    }
//     _net.addNeuron(_currentNeuronIndex+i, v_thresh, tau_refrac, 3.0, 0.0, 0.2, 0.2, 0.2);
    _net.addNeuron (_mIF_type, _currentNeuronIndex+i, 13, args);
    
    bool isExcitatory = (randRange(0.0, 100.0)>(100.0-excitatoryPercent));
    _excitatoryConnections.push_back(isExcitatory);
  }      
  _currentNeuronIndex 	+= newLayer._nNeurons;
  _nNeurons 		+= newLayer._nNeurons;
  _layers.push_back(newLayer);     
}

void LSM::addInputLayer( int x, int y, int z, unsigned int w, unsigned int h, unsigned int d, double excitatoryPercent){
  InputLayer newLayer(_currentNeuronIndex, x, y, z, w, h, d);
  
  for( unsigned i = 0; i < newLayer._nNeurons; i++ ) {
    static float args[1];
//     args[0] = 0.0;
    _net.addNeuron (_mInput_type, _currentNeuronIndex+i, 0, args);

    bool isExcitatory = (randRange(0.0, 100.0)>(100.0-excitatoryPercent));
    _excitatoryConnections.push_back(isExcitatory);    
  }      
  
  _currentNeuronIndex 	+= newLayer._nNeurons;
  _nNeurons 		+= newLayer._nNeurons;
  _inputLayers.push_back(newLayer);     
}

void LSM::addPoissonLayer( int x, int y, int z, unsigned int w, unsigned int h, unsigned int d, double excitatoryPercent) {
  PoissonLayer newLayer(_currentNeuronIndex, x, y, z, w, h, d);
  
  for( unsigned i = 0; i < newLayer._nNeurons; i++ ) {
    float rate = 0.0f;
    _net.addNeuron (_mPoisson_type, _currentNeuronIndex+i, 1, &rate);
    
    bool isExcitatory = (randRange(0.0, 100.0)>(100.0-excitatoryPercent));
    _excitatoryConnections.push_back(isExcitatory);    
  }      
  
  _currentNeuronIndex 	+= newLayer._nNeurons;
  _nNeurons 		+= newLayer._nNeurons;
  _poissonLayers.push_back(newLayer);  
}

void LSM::connectLayerGeneric( GeneralLayer &fromLayerS,       GeneralLayer &toLayerS,
				unsigned int nConnections,  	
				unsigned int lambda, 	      	      double distanceMetric,
				      double weightMean, 	      double weightVariance,
					bool bLearning  ) {
  int nConnected = 0;
  while (nConnected < nConnections ) {
      unsigned neuron1 = randRange( fromLayerS._minIndex, fromLayerS._maxIndex-1);
      unsigned neuron2 = randRange( toLayerS._minIndex,   toLayerS._maxIndex-1);            

      Eigen::Vector3i pos1 = fromLayerS.getNeuronPosition(neuron1);
      Eigen::Vector3i pos2 = toLayerS.getNeuronPosition(neuron2);
      Eigen::Vector3i dis  = (pos1-pos2);      
      double distance = dis.norm(); 
                  
      double delay = distance * distanceMetric;                 
      if( delay <= 1 ) delay = 1;
      if( delay > 30 ) delay = 30;       
            
      double C = 0.0;
	   if(  _excitatoryConnections[neuron1] &&  _excitatoryConnections[neuron2] ) C = 0.3;
      else if( !_excitatoryConnections[neuron1] && !_excitatoryConnections[neuron2] ) C = 0.1;
      else if(  _excitatoryConnections[neuron1] && !_excitatoryConnections[neuron2] ) C = 0.4;
      else if( !_excitatoryConnections[neuron1] &&  _excitatoryConnections[neuron2] ) C = 0.2;
      
      double disFactor = distance / lambda;
      double probabilityOfConnection = C * exp(-pow(disFactor,2));
      
      if( randRange(0.0, 100.0) > probabilityOfConnection ) {
	 double randWeight = (_excitatoryConnections[neuron1]?1:-1) * generateGaussian(weightMean, weightVariance);
	_net.addSynapse (neuron1 , neuron2 , delay , randWeight, bLearning );
	out(string_format("Created connection  %u with %u with delay %lf, weight %lf", neuron1, neuron2, delay, randWeight));
	
	nConnected++;
      };
  }        
  out(string_format("Created %d connections", nConnected));
}

void LSM::connectLayer( unsigned int fromLayer,  	unsigned int toLayer, 
		        unsigned int nConnections,  	
		        unsigned int lambda,     	      double distanceMetric,
		              double weightMean, 	      double weightVariance,
		     	        bool bLearning ) { 
  connectLayerGeneric(_layers[fromLayer], _layers[toLayer], nConnections, 
		       lambda, distanceMetric, weightMean, weightVariance, bLearning );
}  

void LSM::interConnectLayer( unsigned int layer,  		
			     unsigned int nConnections,  	
			     unsigned int lambda,  		      double distanceMetric,
			 	   double weightMean, 	      	      double weightVariance,
				     bool bLearning ) { 
  connectLayerGeneric(_layers[layer], _layers[layer], nConnections, 
		       lambda, distanceMetric, weightMean, weightVariance, bLearning );
}  

void LSM::connectInput 	( unsigned int fromLayer,  	unsigned int toLayer, 
			  unsigned int nConnections,  	
			  unsigned int lambda, 	      	      double distanceMetric,
			        double weightMean, 	      double weightVariance,
				  bool bLearning ) { 
  connectLayerGeneric(_inputLayers[fromLayer], _layers[toLayer], nConnections, 
		       lambda, distanceMetric, weightMean, weightVariance, bLearning );
}

void LSM::connectPoisson ( unsigned int fromLayer,  	unsigned int toLayer, 
			  unsigned int nConnections,  	
			  unsigned int lambda, 	     	      double distanceMetric,
			        double weightMean, 	      double weightVariance,
				  bool bLearning ) {
  connectLayerGeneric(_poissonLayers[fromLayer], _layers[toLayer], nConnections, 
		       lambda, distanceMetric, weightMean, weightVariance, bLearning ); 
}

void LSM::connectPoissonAll (       double connectionPercentage, 
			      unsigned int fromLayer,  		unsigned int toLayer, 
			            double weightMean, 	              double weightVariance,
			    	      bool bLearning ) {
  for( unsigned i = 0; i < _poissonLayers[fromLayer]._nNeurons; i++ ) {
    for( unsigned j = 0; j < _layers[toLayer]._nNeurons; j++ ) {
      if( randRange(0.0, 100.0) > (100.0 - connectionPercentage) ) {
	double randWeight = (_excitatoryConnections[_poissonLayers[fromLayer]._minIndex + i]?1:-1) * 
				  generateGaussian(weightMean, weightVariance);
	_net.addSynapse (_poissonLayers[fromLayer]._minIndex + i , 
			_layers[toLayer]._minIndex + j , 1.0, (float)randWeight, bLearning);
      }
    }           
  }
}

LSM::LSM() 
: /*SecClass(),*/
  _simTime(0),
  _currentNeuronIndex(0),
  _nNeurons(0),
  _outMode(outCout) {   
    _outfile.open("/home/manos/Development/Projects/Nemo/TestNemo/build/lsm.txt", std::ios_base::in);//app);
    _mPoisson_type 	= _net.addNeuronType("PoissonSource");
    _mInput_type   	= _net.addNeuronType("Input");
    _mIF_type 		= _net.addNeuronType("IF_curr_exp");
}

LSM::~LSM() {
//   _sim->_sim->~Simulation();
//   _sim->_sim->~ReadableNetwork();

//   nemo_delete_simulation(_sim);
  
  _layers.clear();
  _inputLayers.clear();
  _poissonLayers.clear();
  
  
  _fired.clear();
  for( unsigned i = 0; i < _firings.size(); i++ ) {
    _firings[i].clear();
  }
  
}

void LSM::init(NemoSim *sim) {
 
//   _conf.setCudaPartitionSize (1024);
//   _conf.setCudaBackend();
 _conf.setCpuBackend();
  _sim = sim;
//  _sim.reset(nemo::simulation(_net, _conf));
 
 _spikeFirings.resize(_nNeurons, 1);
}

void LSM::priorToSim	() {
 for( unsigned i = 0; i < _inputLayers.size(); i++ ) {
    _inputLayers[i].clear();
 }
 for( unsigned i = 0; i < _poissonLayers.size(); i++ ) {
    _poissonLayers[i].clear();
 } 
}

//Parses all network output and assigns smaller values to spikes passed more
Eigen::MatrixXd LSM::filter(unsigned int layerIndex) {
  Eigen::MatrixXd layerFirings = _layers[layerIndex].makeMatrix(_firings);
  
  for( unsigned i = 0; i < layerFirings.rows(); i++ ) {
   unsigned int last_firing = -999999;
   for( unsigned j = 0; j < layerFirings.cols(); j++ ) {
      if( layerFirings(i,j)==1 ) last_firing = j;
      layerFirings(i,j) = (last_firing==-999999)? 0 : 1.0/filterFun(j - last_firing);
   }
  }
  
  return layerFirings;
}

double LSM::filterFun(double inValue) {
 return exp(inValue); 
}

const std::vector<unsigned> LSM::simulate(unsigned int nsteps) {
  _simTime ++;
  
  nemo::Simulation::firing_stimulus 	firing_stimulus;	//This is a vector of neurons forced to fire
  nemo::Simulation::current_stimulus 	current_stimulus;	//This is a stimulus vector

  for( unsigned i = 0; i < _poissonLayers.size(); i++ ) {
    for( unsigned j = 0; j < _poissonLayers[i]._stimulus_input.size(); j++ ) {
      current_stimulus.push_back (_poissonLayers[i]._stimulus_input[j]);      
    }
  }
  
  for( unsigned i = 0; i < _inputLayers.size(); i++ ) {
    for( unsigned j = 0; j < _inputLayers[i]._forced_to_fire.size(); j++ ) {
      firing_stimulus.push_back (_inputLayers[i]._forced_to_fire[j]);      
    }
  }  
  
  for( unsigned int n = 0; n < nsteps; n++ ) {
    _fired = _sim->_sim->step (firing_stimulus, current_stimulus);
    _firings.push_back(_fired);
  }
  
//   _spikeFirings.conservativeResize(Eigen::NoChange, _spikeFirings.cols()+1);
//   for( unsigned i = 0; i < _fired.size(); i++ ) {
//     _spikeFirings(_fired[i], _simTime) = 1.0;
//   }
//   for ( vector < unsigned >:: const_iterator n = fired.begin (); n != fired.end (); ++ n ) {
//     cout << _sim->elapsedSimulation() << " fired " << * n << endl ;
//   }	
  return _fired;
}