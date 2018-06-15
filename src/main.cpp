/**
 * main.cpp
 * 
 * Written by Dan Adler
 * Email: daadler0309@gmail.com
 * GitHub: https://github.com/dadler6/
 * 
 * File to run the neural network code.  This file will essentially
 * fit too purposes.  It will
 * 
 * 1) Build a neural network, through opening up a specified dataset
 *    (train and target), and fitting a neural network to this data,
 *    and then save the network to a specified file
 * 2) Predict on a network.  This will intake a training dataset,
 *    an already trained network, and then predict information
 *    from that network.
 */

// Include
#include <iostream>
#include <fstream>
#include "NeuralNetwork.hpp"

// Using
using namespace neuralnetwork;

/**
 * Train network using a given dataset, train and target columns.
 * 
 * params:
 * MatrixXf X, an array of features
 * VectorXf y, a vector of target data points
 * string output_filename, the output filename to save neural network too
 */
int train_network(MatrixXf X, VectorXf y, string filename) {

    // Build a neural network
    NeuralNetwork ann = NeuralNetwork();

    // Fit the network
    cout << 'Fitting network...' << endl;
    ann.fit(X, y);

    // Save network
    cout << 'Saving network...' << endl;
    ofstream ofs(filename);
    ofs << ann;
    ofs.close();

    // End
    cout << "Network saved!" << endl;

    return 0;
}

/**
 * Predict values in a neural network based upon an input matrix.
 * 
 * params:
 * MatrixXf X, an array of features
 * string ann_filename, an input filename of the neural network
 * string output_filename, the filename to output the result vector to.
 * 
 * returns:
 * VectorXf, the vector of target values (predicted)
 */
int predict_values(MatrixXf X, string ann_filename, string output_filename) {

    // Load in file
    cout << "Reading in neural network..." << endl;
    ifstream ifs(ann_filename);
    NeuralNetwork ann;
    ifs >> ann;

    // Predict data
    cout << "Predicting data..." << endl;
    VectorXf y = ann.predict(X);

    // End
    cout << "Outputting data..." << endl;
    ofstream ofs(output_filename);
    if (ofs.is_open()) {
        ofs << "PREDICTED" << "\n" << y;
    }
    return 0;
}

/**
 * Runs the program.  Will look for arguments and output errors
 * if arguments do not exist.
 */
int main() {
    cout << "Hello, world!" << endl;
    return 0;
}