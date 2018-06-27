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
NeuralNetwork::NeuralNetwork(
    int n_layers, 
    VectorXf conf, 
    float step, 
    float thresh
) {
    // Set parameters
    num_layers = n_layers;
    eta = step;
    threshold = thresh;
    // Create the vector
    for (int i = 0; i < num_layers; i++) {
        weights.push_back(VectorXf::Random(conf(i)).array() - 0.5);
    };
};


NeuralNetwork::~NeuralNetwork(void) {

};


void NeuralNetwork::fit(MatrixXf X, VectorXf y) {
    // Set target vector to 0
    int curr_iter = 0;
    o = VectorXf::Zero(y.rows());
    // Go through ending criteria
    while ((curr_error > cutoff_err) || (curr_iter < max_iter)) {
        // Back propograte
        // Feed forward
        feed_forward(X, y);
        // Update error
        error_term(o, y);
    }
};


ostream & operator<<(ostream &out, const NeuralNetwork &nn) {
    // Save contents to file
};


istream & operator>>(istream &in, const NeuralNetwork &nn) {
    // Open up file
};


VectorXf NeuralNetwork::predict(MatrixXf X) {
    // For testing purposes
    return NeuralNetwork::error_term(X.col(0), X.col(1));
};


// Private methods
static VectorXf error_term(VectorXf output, VectorXf target) {
    return output.array() * 
           (1 - output.array()).array() * 
           (target - output).array();
};


void NeuralNetwork::feed_forward(MatrixXf X, VectorXf y) {
    // Copy matrix into inputs
    MatrixXf curr = X;
    MatrixXf temp_weights;
    // Go through each part of the vactor and multiply by the weights
    for(
        vector<VectorXf>::iterator it = weights.begin(); 
        it != weights.end(); ++it
    ) {
        // Replicate current value of the iterator into a matrix
        temp_weights = it->replicate(1, curr.rows());
        curr *= temp_weights;
    };
    // Pass through sigmoid to get output
    MatrixXf output = sigmoid(curr);
    // Threshold
};


void NeuralNetwork::sse(VectorXf y, VectorXf target) {
    // Calcualte error and set as curr_error param
    curr_error = ((y - target) * (y - target)).sum();
};


static VectorXf sigmoid(VectorXf output) {
    // Run the sigmoid function
    return 1.0 / (output.exp().array() + 1).array();
};


void NeuralNetwork::threshold_output(VectorXf output) {
    // Threshold based upon the threshold parameter
    VectorXf ones = VectorXf::Ones(output.size());
    VectorXf zeros = VectorXf::Zero(output.size());
    o = (output.array() > threshold).select(ones, zeros);
};

}