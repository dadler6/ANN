/**
 * main.cpp
 * 
 * Written by Dan Adler
 * Email: daadler0309@gmail.com
 * GitHub: https://github.com/dadler6/
 * 
 * Main function for building and/or predicting a neural network.
 */


// Include
#include "DataIO.hpp"

// Using
using namespace std;

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
    if ((strcmp(argv[1], "train") == 0) && (argc == 8)) {
        // Get other parameters
        float eta = stof(argv[4]);
        float thresh = stof(argv[5]);
        int num_layers = stof(argv[6]);
        VectorXi config = open_data(argv[7]).cast<int>();
        // Train data
        output_flag = train_network(
            X, eta, thresh, output_file, num_layers, config
        );
    } else if ((strcmp(argv[1], "predict") == 0) && (argc == 5)) {
        // Get other parameters
        string ann_file = argv[4];
        output_flag = predict_values(X, ann_file, output_file);
    } else {
        output_flag = 1;
    }

    return output_flag;
}
