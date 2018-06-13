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
#include <Eigen/Dense>
#include "../src/NeuralNetwork.cpp"
#include <gtest/gtest.h>

// Practice test

// TEST(ErrorTermCheck, PositiveNos) { 
//     ASSERT_EQ(6, squareRoot(36.0));
//     ASSERT_EQ(18.0, squareRoot(324.0));
//     ASSERT_EQ(25.4, squareRoot(645.16));
//     ASSERT_EQ(0, squareRoot(0.0));
// }