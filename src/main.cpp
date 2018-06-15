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
 * 
 * To train (1) you can use the following command:
 * ./run train input_data output_ann_file step_size
 * 
 * To predict (2) you can use the following command:
 * ./run predict input_data output_data_file ann_file 
 */

// Include
#include <iostream>
#include <fstream>
#include <vector>
#include "NeuralNetwork.hpp"

// Using
using namespace neuralnetwork;

/**
 * Open data based upon an input file name.
 * 
 * WE ASSUME the last column of the data is the variables to predict (target)
 * 
 * params:
 * string filename, the filename to read in
 * 
 * return:
 * MatrixXf, the train + target data
 */
MatrixXf open_data(string filename) {
    // Read in file
    cout << "Reading in input data..." << endl;
    ifstream in;
    in.open(filename);

    // Now make matrix, initialize vector to do so
    vector<float> cells;
    string line;
    
    // Count current lines
    int rows = 0;
    int cols;
    while(getline(in, line)) {
        cols = 0;
        stringstream lineStream(line);
        string cell;
        if (rows > 0) {
            while(getline(lineStream, cell, ',')) {
                cells.push_back(stod(cell));
                cols++;
            }
        }
        rows++;
    }

    // Initialize matrix and return
    MatrixXf X(cells.data());

    return X;
}

/**
 * Train network using a given dataset, train and target columns.
 * 
 * params:
 * MatrixXf X, an array of features where the last column is the target
 * folat eta, the gradient descent step size
 * string output_filename, the output filename to save neural network too
 */
int train_network(MatrixXf X, float eta, string filename) {
    // Get y data from X
    VectorXf y = X.rightCols(1);
    // Drop last column of X
    X = X.leftCols(X.cols() - 1);

    // Build a neural network
    NeuralNetwork ann = NeuralNetwork();

    // Fit the network
    cout << 'Fitting network...' << endl;
    ann.fit(X, y, eta);

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
 * 
 * params:
 * int argc, the number of input arguments
 * char *argv[], an array of input arguments
 */
int main(int argc, char *argv[]) {
    // Output flag
    int output_flag;
    // Open data
    string data_file = argv[2];
    MatrixXf X = open_data(data_file);
    // Get output filename
    string output_file = argv[3];
    
    // Check to see if argv is train or predict
    if (strcmp(argv[1], "train") == 0) {
        // Get other parameters
        float eta = stof(argv[4]);
        // Train data
        output_flag = train_network(X, eta, output_file);
    } else if (strcmp(argv[1], "predict") == 0) {
        // Get other parameters
        string ann_file = argv[4];
        output_flag = predict_values(X, ann_file, output_file);
    } else {
        output_flag = 1;
    }

    return output_flag;
}