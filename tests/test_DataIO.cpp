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
#include "../src/DataIO.hpp"
#include <gtest/gtest.h>

// Using
using namespace std;

// CSFFormat
const static IOFormat CSVFormat(
    StreamPrecision, DontAlignCols, ",", "\n"
);

// Test fixture
class DataIOTestFixture: public::testing::Test { 
    protected:
    // Define variables
    MatrixXf X_input;
    char * filename;
    char * saved_network_filename;

    virtual void SetUp() {
        // Add matrix variables
        X_input.resize(4, 3);
        X_input << 0.0, 0.0, 0.0,
             0.0, 1.0, 1.0,
             1.0, 0.0, 1.0,
             1.0, 1.0, 0.0;
        // Prep filenames
        filename = (char*)("data.txt");
        saved_network_filename = (char*)("ann_test.txt");
        // Save to file
        std::ofstream file(filename);
        if (file.is_open()) {
            file << "x1,x2,y\n" << X_input.format(CSVFormat) << '\n';
        }
    };

    virtual void TearDown() {
        // If files exist, erase files
        remove(filename);
        remove(saved_network_filename);
    };
};


TEST_F(DataIOTestFixture, OpenDataTest) {
    // Test opening the data
    string filename_str = string(filename);
    MatrixXf X_result = open_data(filename_str);

    // Assert equality with original matrix
    ASSERT_TRUE(X_input.isApprox(X_result));
}
