/*
 * CNeuralNet.h
 *
 *  Created on: 26 Dec 2013
 *      Author: benjamin
 */

#ifndef CNEURALNET_H_
#define CNEURALNET_H_
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <cstring>
#include <stdio.h>
#include <stdint.h>
#include <iostream>

typedef unsigned int uint;
class CBasicEA; //forward declare EA class, which will have the power to access weight vectors

class CNeuralNet {
	friend class CBasicEA;
protected:
	void feedForward(const double * const inputs); //you may modify this to do std::vector<double> if you want
	void propagateErrorBackward(const double * const desiredOutput); //you may modify this to do std::vector<double> if you want
	double meanSquaredError(const double * const desiredOutput); //you may modify this to do std::vector<double> if you want
public:
	CNeuralNet(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, double lRate, double mse_cutoff);
	void initWeights();
	void train(const double ** const inputs,const double ** const outputs, uint trainingSetSize); //you may modify this to do std::vector<std::vector<double> > or do boost multiarray or something else if you want
	int classify(const double * const input); //you may modify this to do std::vector<double> if you want
	double getOutput(int index) const;
	virtual ~CNeuralNet();

private:
	int inputSize;
	int hiddenSize;
	int outputSize;
	int inputHiddenSize;
	int hiddenOutputSize;

	double **inputHiddenLayer;
	double **hiddenOutputLayer;

	double *inputLayer; 
	double *hiddenLayer;
	double *outputLayer; 

	double learningRate;
	double mseCutOff;
};

#endif /* CNEURALNET_H_ */
