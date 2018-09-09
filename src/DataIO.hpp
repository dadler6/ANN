/**
 * DataIO.hpp
 * 
 * Written by Dan Adler
 * Email: daadler0309@gmail.com
 * GitHub: https://github.com/dadler6/
 * 
 * Implementation file to help with DataIO.  The file will:
 * 
 * 1) Build a neural network, through opening up a specified dataset
 *    (train and target), and fitting a neural network to this data,
 *    and then save the network to a specified file
 * 2) Predict on a network.  This will intake a training dataset,
 *    an already trained network, and then predict information
 *    from that network.
 * 
 * To train (1) you can use the following command:
 * ./run train input_data output_ann_file step_size threshold num_layers layers_config
 * 
 * To predict (2) you can use the following command:
 * ./run predict input_data output_data_file ann_file 
 */

// Include
#include "NeuralNetwork.hpp"
#include <fstream>


// Using
using namespace neuralnetwork;
using namespace std;


/**
 * Open data based upon an input file name.
 * 
 * WE ASSUME the last column of the data is the variables to predict (target)
 * 
 * params:
 * string f, the filename to read in
 * 
 * return:
 * MatrixXf, the train + target data
 */
MatrixXf open_data(string f);


/**
 * Train network using a given dataset, train and target columns.
 * 
 * params:
 * MatrixXf X, an array of features where the last column is the target
 * float eta, the gradient descent step size
 * float thresh, the threshold for deciding if a class is positive or not
 * string output_filename, the output filename to save neural network too
 * int num_layers, the number of layers
 * VectorXf config,  vector showing the configuration of the network
 */
int train_network(
    MatrixXf X, 
    float eta, 
    float thresh,
    string output_filename,
    int num_layers,
    VectorXi config
);


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
int predict_values(MatrixXf X, string ann_filename, string output_filename);
