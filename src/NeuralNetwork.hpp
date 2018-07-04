/**
 * NeuralNetwork.hpp
 * Written by Dan Adler
 * Email: daadler0309@gmail.com
 * GitHub: https://github.com/dadler6/
 * 
 * Header file for the Neural Network class that will utilize back 
 * propogation and stochastic gradient descent.
 */

// Definitions
#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

// Include
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

// Name space
using namespace Eigen;
using namespace std;

namespace neuralnetwork {

class NeuralNetwork {
    
    public:
        /**
         * Default constructor for the neural network.  Can be used if
         * going to fill in parameters later.
         */
        NeuralNetwork(void);


        /**
         * Constructor for the neural network.  Will setup all the basic
         * parameters needed to train the network.
         * 
         * params:
         * int num_layers, the number of layers in one's network
         * VectorXf config, the number of nodes per layer
         * float eta, the training step size for gradiet descent
         * float threshold, the threshold for classification
         */
        NeuralNetwork(int n_layers, VectorXf conf, float step, float thresh);


        /**
         * Destructor for the neural network
         */
        ~NeuralNetwork(void);


        /**
         * Fit a nerual network based upon the features and target output
         * 
         * params:
         * MatrixXf X, an array of features and datapoints 
         *             (n data points x m features)
         * ArrayXf y, the target values (n data points x 1)
         */
        void fit(MatrixXf X, VectorXf y);


        /**
         * Predict a set of target values based upon a set of features
         * 
         * params:
         * MatrixXf X, an array of features and datapoints 
         *             (n data points x m features)
         * 
         * returns:
         * VectorXf, the predicted values for each (n data points x 1)
         */
        VectorXf predict(MatrixXf X);


        /**
         * Save to file using ostream operator
         * 
         * params:
         * ostream &out: the filename to save to
         * NeuralNetwork &nn: the object to write
         */
        friend ostream & operator<<(ostream &out, const NeuralNetwork &nn);


        /**
         * Open a file using the istream operator.
         * 
         * params:
         * istream &in: the filename to open
         * NeuralNetwork &nn: the object to create
         */
        friend istream & operator>>(istream &in, const NeuralNetwork &nn);

    private:
        // Parameters to be set by a user
        vector<MatrixXf> weights;
        vector<VectorXf> temp_outputs;
        vector<VectorXf> delta;
        int num_layers;
        float eta;
        float threshold;
        VectorXf config;
        VectorXf o;

        // Parameters that are constant
        int max_iter = 1000;
        float cutoff_err = 0.05;
        float curr_error = 1000000.0;


        /**
         * Calculate the error term based upon a target and a given output.  
         * Will follow the vectorized version of the following:
         * delta_k = o_k * (1 - o_k)(t_k - o_k)
         * 
         * params:
         * VectorXf output, an array that is the TARGET value (t_k)
         * VectorXf target, an array that is the OUTPUT value (o_k)
         * 
         * returns:
         * VectorXf, an array of the delta_k values
         */
        static VectorXf error_term(VectorXf output, VectorXf target);


        /**
         * Add ones to an input dataset.
         * 
         * params:
         * MatrixXf X, the original data
         * 
         * returns:
         * MatrixXf X_new, the new data with a column of ones appended
         */
        static MatrixXf add_ones(MatrixXf X);


        /**
         * Feed forward parameters and calculate end error term.
         *
         * params:
         * VectorXf x, a datapoint from the training set
         * 
         * return:
         * float, the current output
         */
        float feed_forward(VectorXf x);


        /**
         * Back propogate to find error terms and update weights
         * 
         * params:
         * VectorXf y, the target values
         */
        void back_propogate(VectorXf y);


        /**
         * Update weights
         */
        void update_weights(void);

        /**
         * Get the current sum of squared errors between target and output
         * 
         * params:
         * VectorXf y, the target values
         * VectorXf o, the current output
         */
        void sse(VectorXf y, VectorXf o);


        /**
         * Sigmoid function = 1/(1 + exp(-y))
         * 
         * params:
         * VectorXf output, the current output of the network
         * 
         * return:
         * VectorXf, each element in the original vector with the
         *           sigmoid taken upon it
         */
        static VectorXf sigmoid(VectorXf output);


        /**
         * Threshold based upon some given value
         * 
         * params:
         * VectorXf output, the output float after passing through
         *               the threshold
         * return:
         * VectorXf, the thresholded object
         */
        VectorXf NeuralNetwork::threshold_output(VectorXf output);
};

} // 

#endif // NEURALNETWORK_H_
