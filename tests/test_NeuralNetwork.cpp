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


TEST_F(LogicalTextFixture, LogicalNotTest) {
    // Define input parameters
    MatrixXf X_not;
    VectorXf y_not;
    VectorXi config_not;

    X_not.resize(2, 1);
    X_not << 0.0,
             1.0;

    y_not.resize(2);
    y_not << 1.0, 0.0;

    config_not.resize(2);
    config_not << 1, 1;

    // Define network
    NeuralNetwork ann_not;
    ann_not = NeuralNetwork(n_layers, config_not, step, thresh);

    // Fit
    ann_not.fit(X_not, y_not);

    // Check weights
    vector<MatrixXf> not_weights = ann_not.get_weights();

    // Assert that there exists only one set of weights
    ASSERT_EQ(not_weights.size(), 1);

    // Due to threshold, the weight should be
    // positive and greater than threshold for w_0,
    // and negative and less than -1
    // for the w_1 weight
    ASSERT_GT(not_weights[0](0), 1.0);
    ASSERT_LT(not_weights[0](0) + not_weights[0](1), 1.0);
};


TEST_F(LogicalTextFixture, LogicalAndTest) { 
    // Define network
    NeuralNetwork ann_and;
    ann_and = NeuralNetwork(n_layers, config, step, thresh);

    // Fit
    ann_and.fit(X, y_and);

    // Check weights
    vector<MatrixXf> and_weights = ann_and.get_weights();

    // Assert that there exists only one set of weights
    ASSERT_EQ(and_weights.size(), 1);

    // Due to threshold (which is when weights sum to 1), check
    // that weights individually are < 1, but together are not
    // less than one
    // Also need to remember that weight(0) is the intercept weight
    // w_0, so need to make sure that that is taken into account
    ASSERT_LT(and_weights[0](0), 1.0);
    ASSERT_LT(and_weights[0](0) + and_weights[0](1), 1.0);
    ASSERT_LT(and_weights[0](0) + and_weights[0](2), 1.0);
    ASSERT_GT(and_weights[0].sum(), 1.0);
}


TEST_F(LogicalTextFixture, LogicalOrTest) { 
    // Define network
    NeuralNetwork ann_or;
    ann_or = NeuralNetwork(n_layers, config, step, thresh);

    // Fit
    ann_or.fit(X, y_or);

    // Check weights
    vector<MatrixXf> or_weights = ann_or.get_weights();

    // Assert that there exists only one set of weights
    ASSERT_EQ(or_weights.size(), 1);

    // Due to threshold (which is when weights sum to 1), check
    // that weights individually are > 1,
    // Also need to remember that weight(0) is the intercept weight
    // w_0, so need to make sure that that is taken into account
    // when summing weights
    ASSERT_LT(or_weights[0](0), 1.0);
    ASSERT_GT(or_weights[0](0) + or_weights[0](1), 1.0);
    ASSERT_GT(or_weights[0](0) + or_weights[0](2), 1.0);
}

 TEST_F(LogicalTextFixture, LogicalXOrTest) { 
    // Define output
    int n_layers_2 = 3;
    VectorXi config_2;
    config_2.resize(3);
    config_2 << 2, 2, 1;

    // Define network
    NeuralNetwork ann_xor;
    ann_xor = NeuralNetwork(n_layers_2, config_2, step, thresh);

    // Fit
    ann_xor.fit(X, y_xor);

    // Check weights
    vector<MatrixXf> xor_weights = ann_xor.get_weights();

    // Assert that there exists only one set of weights
    ASSERT_EQ(xor_weights.size(), 2);

    // Cannot easily test since multi-layer means
    // non-linearity.  Simply going to test the predict
    // function in this case with a vector of data
    // This is a little silly since I am predicting on
    // the training data essentially, but at least will
    // show a proof of concept with the predict function.
    VectorXf pred_y = ann_xor.predict(X);
    ASSERT_EQ(pred_y(0), 0.0);
    ASSERT_EQ(pred_y(1), 1.0);
    ASSERT_EQ(pred_y(2), 1.0);
    ASSERT_EQ(pred_y(3), 0.0);
}