/*
                                                                           
   (               )                                        )              
 ( )\     )     ( /(       (                  (  (     ) ( /((             
 )((_) ( /(  (  )\())`  )  )(   (  `  )   (   )\))( ( /( )\())\  (   (     
((_)_  )(_)) )\((_)\ /(/( (()\  )\ /(/(   )\ ((_))\ )(_)|_))((_) )\  )\ )  
 | _ )((_)_ ((_) |(_|(_)_\ ((_)((_|(_)_\ ((_) (()(_|(_)_| |_ (_)((_)_(_/(  
 | _ \/ _` / _|| / /| '_ \) '_/ _ \ '_ \/ _` |/ _` |/ _` |  _|| / _ \ ' \)) 
 |___/\__,_\__||_\_\| .__/|_| \___/ .__/\__,_|\__, |\__,_|\__||_\___/_||_|  
                    |_|           |_|         |___/                         

 For more information on back-propagation refer to:
 Chapter 18 of Russel and Norvig (2010).
 Artificial Intelligence - A Modern Approach.
 */

#include "CNeuralNet.h"

/**
 The constructor of the neural network. This constructor will allocate memory
 for the weights of both input->hidden and hidden->output layers, as well as the input, hidden
 and output layers.
*/
CNeuralNet::CNeuralNet(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, double lRate, double mse_cutoff) :
inputSize(inputLayerSize), hiddenSize(hiddenLayerSize), outputSize(outputLayerSize), learningRate(lRate), mseCutOff(mse_cutoff)
{	
	std::cout << "const = " << hiddenSize << std::endl;
	//sizes for the inbetween layers
	inputHiddenSize = inputSize*hiddenSize;
	hiddenOutputSize = hiddenSize*outputSize;

	//inbetween layers
	inputHiddenLayer = new double*[inputSize];
	hiddenOutputLayer = new double*[hiddenSize];
	
	//input, hiddena and output layers
	inputLayer = new double[inputSize];
	hiddenLayer = new double[hiddenSize];
	outputLayer = new double[outputSize];


	//intialising the inputHidden layer:: for each input node, create an array of doubles that correspong to every hidden node
	for (int i = 0; i < inputSize; i++)
	{
		inputHiddenLayer[i] = new double[hiddenSize];
	}

	//initialising the hiddenOutput layer:: for each hidden node, creat an array of doubles that correspond to every output node
	for (int j = 0; j < hiddenSize; j++)
	{
		hiddenOutputLayer[j] = new double[outputSize];
	}

	initWeights();
	std::cout << "const end" << std::endl;
}
/**
 The destructor of the class. All allocated memory will be released here
*/
CNeuralNet::~CNeuralNet() 
{
	
	for (int i = 0; i < inputHiddenSize; i++)
	{
		delete[] inputHiddenLayer[i];
	}
	delete[] inputHiddenLayer;
	
	for (int i = 0; i < hiddenOutputSize; i++)
	{
		delete[] hiddenOutputLayer[i];
	}
	delete[] hiddenOutputLayer;
	delete[] outputLayer;
	delete[] hiddenLayer;
	delete[] inputLayer;
}
/**
 Method to initialize the both layers of weights to random numbers
*/
void CNeuralNet::initWeights()
{	
	int max = 1;
	int min = -1;
	//initialising inputHiddenLayer values to random numbers
	for (int i = 0; i < hiddenSize; i++)
	{
		for (int j = 0; j < inputSize; j++)
		{
			inputHiddenLayer[j][i] = ((double)rand() / ((RAND_MAX) / 2)) - 1;
			//std::cout << ((double)rand() / ((RAND_MAX) / 2)) - 1 << std::endl;
		}
	}

	//initialising hiddenOutputLayer values to random numberss
	for (int i = 0; i < outputSize; i++)
	{
		for (int j = 0; j < hiddenSize; j++)
		{
			hiddenOutputLayer[j][i] = ((double)rand() / ((RAND_MAX) / 2)) - 1;
		}
	}
}
/**
 This is the forward feeding part of back propagation.
 1. This should take the input and copy the memory (use memcpy / std::copy)
 to the allocated _input array.
 2. Compute the output of at the hidden layer nodes 
 (each _hidden layer node = sigmoid (sum( _weights_h_i * _inputs)) //assume the network is completely connected
 3. Repeat step 2, but this time compute the output at the output layer
*/
void CNeuralNet::feedForward(const double * const inputs) 
{	
	//std::cout << "start feedforward" << std::endl;
	//step one
	memcpy(inputLayer, inputs, inputSize * sizeof(double));

	//step 2: computing the output of the hidden layer nodes
	double hiddenSum = 0;
	for (int i = 0; i < hiddenSize; i++)
	{
		hiddenSum = 0;
		for (int j = 0; j < inputSize; j++)
		{
			hiddenSum += inputHiddenLayer[j][i] * inputs[j];
			//std::cout << inputHiddenLayer[j][i] << std::endl;
		}
		double hiddenLayerOutput = 1 / (1 + exp(-hiddenSum));
		hiddenLayer[i] = hiddenLayerOutput;
	}

	//step 3: computing output for the output layer
	double outputSum = 0;
	for (int i = 0; i < outputSize; i++)
	{
		outputSum = 0;
		for (int j = 0; j < hiddenSize; j++)
		{
			outputSum += hiddenOutputLayer[j][i] * hiddenLayer[j];
		}
		double outputLayerOutput = 1 / (1 + exp(-outputSum));
		outputLayer[i] = outputLayerOutput;
	}
	//std::cout << "end feedforward" << std::endl;
	
}
/**
 This is the actual back propagation part of the back propagation algorithm
 It should be executed after feeding forward. Given a vector of desired outputs
 we compute the error at the hidden and output layers (allocate some memory for this) and
 assign 'blame' for any error to all the nodes that fed into the current node, based on the
 weight of the connection.
 Steps:
 1. Compute the error at the output layer: sigmoid_d(output) * (difference between expected and computed outputs)
    for each output
 2. Compute the error at the hidden layer: sigmoid_d(hidden) * 
	sum(weights_o_h * difference between expected output and computed output at output layer)
	for each hidden layer node
 3. Adjust the weights from the hidden to the output layer: learning rate * error at the output layer * input to output node
    for each connection between the hidden and output layers
 4. Adjust the weights from the input to the hidden layer: learning rate * error at the hidden layer * input layer node value
    for each connection between the input and hidden layers
 5. REMEMBER TO FREE ANY ALLOCATED MEMORY WHEN YOU'RE DONE (or use std::vector ;)
*/


void CNeuralNet::propagateErrorBackward(const double * const desiredOutput)
{
	//std::cout << "begin backprop" << std::endl;
	//step 1: compute error at the output layer
	double *outputLayerErrors = new double[outputSize];
	//std::cout << "begin backprop" << std::endl;
		for (int j = 0; j < outputSize; j++)
		{
			double d_sigmoid = outputLayer[j]*(1-outputLayer[j]);
			double error = d_sigmoid * (desiredOutput[j] - outputLayer[j]);

			outputLayerErrors[j] = error;
		}
	//std::cout << "s1 end = " << hiddenSize << std::endl;

	//step 2: compute the error at the hidden layer
	double  * hiddenLayerErrors = new double[hiddenSize];

	//std::cout << "beginging step 2" << std::endl;
	for (int i = 0; i < hiddenSize; i++)
	{
		//std::cout << "setting error" << std::endl;
		double error = 0.0f;
		for (int j = 0; j < outputSize; j++)
		{
			//not too sure about this step here
			error += hiddenOutputLayer[i][j]*(outputLayerErrors[j]);
		}
		hiddenLayerErrors[i] = (hiddenLayer[i] * (1 - hiddenLayer[i]))*error;
	}

	//std::cout << "s2 end" << std::endl;

	//step 3: adjust weights from hidden to output layer
	for (int i = 0; i < outputSize; i++)
	{
		for (int j = 0; j < hiddenSize; j++)
		{
			hiddenOutputLayer[j][i] += learningRate * outputLayerErrors[i] * hiddenLayer[j];
		}
	}
	//std::cout << "s3 end" << std::endl;

	//step 4: adjust weights from input to hidden layer
	for (int i = 0; i < hiddenSize; i++)
	{
		for (int j = 0; j < inputSize; j++)
		{
			inputHiddenLayer[j][i] += learningRate * hiddenLayerErrors[i] * inputLayer[j];
			//std::cout << hiddenLayerErrors[i] << std::endl;
		}
	}
	//std::cout << "s4 end" << std::endl;

	
	delete[] outputLayerErrors;
	delete[] hiddenLayerErrors;

	//std::cout << "end backprop" << std::endl;
}
/**
This computes the mean squared error
A very handy formula to test numeric output with. You may want to commit this one to memory
*/
double CNeuralNet::meanSquaredError(const double * const desiredOutput)
{
	/*TODO:
	sum <- 0
	for i in 0...outputLayerSize -1 do
		err <- desiredoutput[i] - actualoutput[i]
		sum <- sum + err*err
	return sum / outputLayerSize
	*/
	
	double sum = 0;
	
	for (int j = 0;  j<outputSize; j++)
	{
		double error = desiredOutput[j] - outputLayer[j];
		sum += error*error;
		//std::cout << desiredOutput[j] << std::endl;
		//std::cout << outputLayer[j] << std::endl;
	}
	return sum/outputSize;
}
/**
This trains the neural network according to the back propagation algorithm.
The primary steps are:
for each training pattern:
  feed forward
  propagate backward
until the MSE becomes suitably small
*/

void CNeuralNet::train(const double** const inputs,
		const double** const outputs, uint trainingSetSize) 
{
	double averageMSE =0.0f;
	double sumMSE =0.0f;
	do
	{
		sumMSE = 0;
		for (int i = 0; i < trainingSetSize; i++)
		{
			feedForward(inputs[i]);
			propagateErrorBackward(outputs[i]);
			sumMSE += meanSquaredError(outputs[i]);
		}
		
		std::cout << "msea = " << sumMSE << std::endl;
	} while (sumMSE > mseCutOff);
	//std::cout << "end train" << std::endl;
}
/**
Once our network is trained we can simply feed it some input though the feed forward
method and take the maximum value as the classification
*/
int CNeuralNet::classify(const double * const input){
	feedForward(input);

	double max = -DBL_MAX;

	if (getOutput(0) > getOutput(1))
	{
		return 0;
	}
	else
	{
		return 1;
	}
}
/**
Gets the output at the specified index
*/
double CNeuralNet::getOutput(int index) const{
	return outputLayer[index];
}

