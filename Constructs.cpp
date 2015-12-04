#include "Constructs.h"

int randRange(int min, int max) {
  int ran = min + (rand() % (int)(max - min + 1));
//   std::cout << "min is: " << min << " max is: " << max << " random is: " << ran << endl;  
  
  return  ran;
}

typedef boost::minstd_rand base_generator_type;
double randRange(double min, double max) {
  base_generator_type generator(42u);  
  srand(rand());
  generator.seed(static_cast<unsigned int>(rand()));
  // Define a uniform random number distribution which produces "double"
  // values between 0 and 1 (0 inclusive, 1 exclusive).
  boost::uniform_real<> uni_dist(0,1);
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);
  
  double ran = min + (uni()*(max-min));
//   std::cout << "min is: " << min << " max is: " << max << " random is: " << ran << " uni is: " << uni() << endl;
  return  ran;
}

unsigned int randRange(unsigned int min, unsigned int max) {
  unsigned int ran = min + (rand() % (int)(max - min + 1));
//   std::cout << "min is: " << min << " max is: " << max << " random is: " << ran << endl;  
  
  return  ran;
}

std::string string_format(const std::string fmt_str, ...) {
    int final_n, n = ((int)fmt_str.size()) * 2; /* Reserve two times as much as the length of the fmt_str */
    std::string str;
    std::unique_ptr<char[]> formatted;
    va_list ap;
    while(1) {
        formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
        strcpy(&formatted[0], fmt_str.c_str());
        va_start(ap, fmt_str);
        final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
        va_end(ap);
        if (final_n < 0 || final_n >= n)
            n += abs(final_n - n + 1);
        else
            break;
    }
    return std::string(formatted.get());
}

//this assumes an std deviation as percentage of mean
//i.e. deviation = 50 means that std's will deviate from mean 
//as 50%
double generateGaussian(double mean, double deviation) {
  double std_deviation = abs((deviation/100.0)*mean) ;
  boost::mt19937 *rng = new boost::mt19937();
  
  srand(rand());  
  rng->seed(rand());
  
  boost::normal_distribution<> distribution(mean, std_deviation);
  boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);
  
  delete rng;
  return dist();
}

GeneralLayer::GeneralLayer(unsigned int minIndex, int x, int y, int z, unsigned int w, unsigned int h, unsigned int d) 
: _nNeurons(w*h*d),
  _minIndex(minIndex),
  _maxIndex(minIndex+_nNeurons),
  _x(x), _y(y), _z(z),
  _w(w), _h(h), _d(d) {
  
}

void GeneralLayer::clear() {
  _stimulus_input.clear();
  _forced_to_fire.clear();
}

Eigen::VectorXd GeneralLayer::makeVector(std::vector<unsigned> firings) {
  VectorXd neuronFirings = VectorXd::Zero(_nNeurons);
  
  for( unsigned i = 0; i < firings.size(); i++ ) {
    if( checkNeuron(firings[i]) ) {
	neuronFirings(firings[i]-_minIndex) = 1.0;
    }
  }
  
  return neuronFirings;
}

Eigen::MatrixXd GeneralLayer::makeMatrix(std::vector< std::vector<unsigned> > allFirings) {
  Eigen::MatrixXd ret(_nNeurons, allFirings.size());
  
  for( unsigned i = 0; i < allFirings.size(); i++ ) {
    Eigen::VectorXd thisFirings = makeVector( allFirings[i] );
    ret.col(i) = thisFirings;
  }
  return ret;
}

bool GeneralLayer::checkNeuron(unsigned int neuronIndex) {
    return (neuronIndex >= _minIndex) && (neuronIndex < _maxIndex);
}

unsigned int GeneralLayer::getNeuronIndexInLSM(unsigned int neuronIndex) {
    return neuronIndex + _minIndex;
}

Eigen::Vector3i GeneralLayer::getNeuronPosition(unsigned int neuronIndex) {
  unsigned int nPosZ = ceil(neuronIndex / (_w * _h));
  unsigned int posInLayer = neuronIndex % (_w * _h);
  
  unsigned int nPosY = ceil(posInLayer/_w);
  unsigned int nPosX = posInLayer % _w;
  
  Eigen::Vector3i position;
  position << nPosX, nPosY, nPosZ;
  return position;
}

void TicToc::tic() {
    tictoc_stack.push(clock());
}

void TicToc::toc(bool bOut) {
  if( bOut ) {
    std::cout << "Time elapsed: "
	      << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
	      << std::endl;
  }
    tictoc_stack.pop();
}  



PoissonLayer::PoissonLayer(unsigned int minIndex, 
			   int x, int y, int z, 
			   unsigned int w, unsigned int h, unsigned int d)
: GeneralLayer(minIndex, x, y, z, w, h, d) {
  
}

void PoissonLayer::inputNeuron(LSM *net, unsigned int neuronIndex, float input) {
//   _stimulus_input.push_back(std::make_pair<unsigned, float>(_minIndex+neuronIndex, input));	 
//   float locInput[1];
//   locInput[0] = input;
//   net->setNeuron(_minIndex+neuronIndex, 1, locInput);
  net->_sim->_sim->setNeuron(_minIndex+neuronIndex, 1, &input);
//   net->setNeuronParameter(_minIndex+neuronIndex, 0, input);
}

InputLayer::InputLayer(unsigned int minIndex, 
		       int x, int y, int z, 
		       unsigned int w, unsigned int h, unsigned int d)
: GeneralLayer(minIndex, x, y, z, w, h, d) {
  
}

void InputLayer::fireNeuron(unsigned int neuronIndex) {
  _forced_to_fire.push_back(neuronIndex);
}