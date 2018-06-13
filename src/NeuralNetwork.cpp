/**
 * NeuralNetwork.cpp
 * 
 * Written by Dan Adler
 * Email: daadler0309@gmail.com
 * GitHub: https://github.com/dadler6/
 * 
 * Implementation file for the Neural Network class that will utilize back 
 * propogation and stochastic gradient descent.
 */

// Include
#include "NeuralNetwork.hpp"

// Namespace
namespace neuralnetwork {


// Public methods
NeuralNetwork::NeuralNetwork(void) {

};

NeuralNetwork::~NeuralNetwork(void) {

};

void NeuralNetwork::fit(MatrixXf X, VectorXf y) {

};

VectorXf NeuralNetwork::predict(MatrixXf X) {
    // For testing purposes
    return NeuralNetwork::error_term(X.col(0), X.col(1));
};

// Private methods
VectorXf NeuralNetwork::error_term(VectorXf output, VectorXf target) {
    return output.array() * (1 - output.array()).array() * (target - output).array();
};

}