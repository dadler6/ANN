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
NeuralNetwork::NeuralNetwork(void) = default;


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
    // Create the vector of matrix weights
    // and add one for the w_o node at each step
    for (int i = 0; i < (num_layers - 1); i++) {
        weights.push_back(
            MatrixXf::Random(conf(i + 1) + 1, conf(i) + 1).array() - 
            0.5
        );
    };
};


NeuralNetwork::~NeuralNetwork(void) {};


void NeuralNetwork::fit(MatrixXf X, VectorXf y) {
    // Set starting parameters
    VectorXf x;
    int curr_iter = 0;
    // Add column of one's to X to work with hidden layers
    MatrixXf X_new = add_ones(X);
    // Go through ending criteria
    while ((curr_error > cutoff_err) || (curr_iter < max_iter)) {
        // Initialize new output to 0
        o = VectorXf::Zero(y.rows());
        // Go through each training example
        for (size_t r = 0; r < X_new.rows(); r++) {
            // Get current row
            x = X_new.row(r);
            // Feed forward
            o(r) = feed_forward(x);
            // Back propogate
            back_propogate(y);
            // Update weights
            update_weights();
        };
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
    // Add ones
    MatrixXf X_new = add_ones(X);
    // Initialize vector of 0's as output
    o = VectorXf::Zero(X_new.rows());
    // Go through each traning example
    for (size_t r = 0; r < X_new.rows(); r++) {
        o(r) = feed_forward(X_new.row(r));
    }
    // For testing purposes
    return o;
};


// Private methods
static VectorXf error_term(VectorXf output, VectorXf target) {
    return output.array() * 
           (1 - output.array()).array() * 
           (target - output).array();
};

static MatrixXf add_ones(MatrixXf X) {
    // Add row of ones
    MatrixXf X_new = MatrixXf::Ones(X.rows(), X.cols() + 1);
    X_new.block(0, 1, X.rows(), X.cols()) = X;
    return X_new;
};


float NeuralNetwork::feed_forward(VectorXf x) {
    // Copy matrix into inputs
    VectorXf curr = x;

    // Get a temporary output
    VectorXf temp_output;

    // Clear vector outputs
    temp_outputs.clear();

    // Go through each part of the vactor and multiply by the weights
    for(
        vector<MatrixXf>::iterator it = weights.begin(); 
        it != weights.end(); 
        ++it
    ) {
        // Push back current value
        temp_outputs.push_back(curr);
        // Replicate current value of the iterator into a matrix
        curr = sigmoid(*it * curr);
    };
    // Pass through threshold to get output
    temp_output = threshold_output(curr);
    temp_outputs.push_back(temp_output);
    return temp_output(0);
};


void NeuralNetwork::sse(VectorXf y, VectorXf target) {
    // Calcualte error and set as curr_error param
    curr_error = ((y - target) * (y - target)).sum();
};


static VectorXf sigmoid(VectorXf output) {
    // Run the sigmoid function
    return 1.0 / (output.exp().array() + 1).array();
};


VectorXf NeuralNetwork::threshold_output(VectorXf output) {
    // Threshold based upon the threshold parameter
    VectorXf ones = VectorXf::Ones(output.size());
    VectorXf zeros = VectorXf::Zero(output.size());
    return (output.array() > threshold).select(ones, zeros);
};


void NeuralNetwork::back_propogate(VectorXf y) {
    // Clear vector and set end counter
    delta.clear();
    bool end_counter = true;
    int curr_pos = 0;

    // Start at the end of the weights and calculate errors
    for (
        vector<VectorXf>::reverse_iterator i = temp_outputs.rbegin(); 
        i != temp_outputs.rend(); 
        i++
    ) {
        // Check if at the end of the vector
        if (end_counter) {
            // Calculate:
            // delta_i = o * (1 - o) * (t - o)
            delta.push_back(
                i->array() * 
                (1 - i->array()) * (y.array() - i->array())
            );
            end_counter = false;
        } else {
            // Calculate:
            // delta_i = o * (1 - o) * W * delta_{i + 1}
            delta.push_back(
                i->array() * 
                (1 - i->array()) * 
                (
                    weights[num_layers - 1 - curr_pos] * 
                    delta[curr_pos]
                ).array()
            );
            curr_pos++;
        };
    };
};


void NeuralNetwork::update_weights(void) {
    int curr_pos = 0;
    // Go through each weight set and update
    for (
        vector<VectorXf>::iterator i = temp_outputs.begin(); 
        i != temp_outputs.end(); 
        i++
    ) {
        // Update weights
        weights[curr_pos] += eta * delta[curr_pos] * (*i);
        curr_pos++;
    };
};

}
