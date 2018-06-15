/**
 * test_main.cpp
 * 
 * Written by Dan Adler
 * Email: daadler0309@gmail.com
 * GitHub: https://github.com/dadler6/
 * 
 * Tets the code that:
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
#include "../src/main.cpp"
#include <gtest/gtest.h>

// Using
using namespace neuralnetwork;
using namespace std;

/**
 * TestOpenData
 * 
 * Tests the funtion that opens data and loads it into an eigen matrix
 */
TEST(TestOpenData, openCSV) {
    //
}