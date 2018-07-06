/**
 * test_NeuralNetwork.cpp
 * 
 * Written by Dan Adler
 * Email: daadler0309@gmail.com
 * GitHub: https://github.com/dadler6/
 * 
 * Tests the code that is an:
 * Implementation file for the Neural Network class that 
 * will utilize back propogation and stochastic gradient descent.
 */

// Include
#include "../src/NeuralNetwork.hpp"
#include <gtest/gtest.h>
#include <cmath>

// Using
using namespace neuralnetwork;
using namespace std;

// Test fixture
class LogicalTextFixture: public::testing::Test { 
    protected:
        // Define variables
        MatrixXf X;
        VectorXf y_and;
        VectorXf y_or;
        VectorXf y_xor;
        int n_layers;
        VectorXi config;
        float step;
        float thresh;
        
        virtual void SetUp() { 
            // Initialize input/target data
            X.resize(4, 2);
            X << 0.0, 0.0, 
                 0.0, 1.0, 
                 1.0, 0.0, 
                 1.0, 1.0;

            y_and.resize(4);
            y_and << 0.0, 0.0, 0.0, 1.0;

            y_or.resize(4);
            y_or << 0.0, 1.0, 1.0, 1.0;

            y_xor.resize(4);
            y_xor << 0.0, 1.0, 1.0, 0.0;

            // Initialize other parameters
            n_layers = 2;
            config.resize(2);
            config << 2, 1;
            step = 0.1;
            thresh = 1.0 / (1.0 + exp(-1));
        };
};

TEST_F(LogicalTextFixture, LogicalAndTest) { 
    // Define inputs
    NeuralNetwork ann_and;
    ann_and = NeuralNetwork(n_layers, config, step, thresh);

    // Fit
    ann_and.fit(X, y_and);

    // Check weights
    vector<MatrixXf> and_weights = ann_and.get_weights();

    // Assert that there exists only one set of weights
    ASSERT_EQ(and_weights.size(), 1);
}

TEST_F(LogicalTextFixture, LogicalOrTest) { 
    // Define inputs
    NeuralNetwork ann_or;
    ann_or = NeuralNetwork(n_layers, config, step, thresh);

    // Fit
    ann_or.fit(X, y_or);

    // Check weights
    vector<MatrixXf> or_weights = ann_or.get_weights();

    // Assert that there exists only one set of weights
    ASSERT_EQ(or_weights.size(), 1);
 }