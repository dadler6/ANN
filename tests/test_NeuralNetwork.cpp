/**
 * test_NeuralNetwork.cpp
 * 
 * Written by Dan Adler
 * Email: daadler0309@gmail.com
 * GitHub: https://github.com/dadler6/
 * 
 * Tests the code that is an:
 * Implementation file for the Neural Network class that will utilize back 
 * propogation and stochastic gradient descent.
 */

// Include
#include "../src/NeuralNetwork.hpp"
#include <gtest/gtest.h>

// Using
using namespace neuralnetwork;
using namespace std;

// Practice test

TEST(ErrorTermCheck, Check) { 
    // Define inputs
    Matrix2f X_input;
    NeuralNetwork ann;
    X_input << 1, 2, 3, 4;

    // Define result
    Vector2f res_1;
    res_1 = ann.predict(X_input);

    // Define true value
    Vector2f true_1;
    true_1 << 0, -6;

    // Assertions
    ASSERT_TRUE(true_1.isApprox(res_1));
 }