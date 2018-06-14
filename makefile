# Makefile for neural network implementation
#
# Written by Dan Adler
# Email: daadler0309@gmail.com
# GitHub: https://github.com/dadler6/
#
#

# C++ flags
CC = g++
CFLAGS = -std=c++11 -stdlib=libc++ -pedantic -Wall -Wextra -O

# Folders
SRC = ./src
BIN = ./bin
TESTS = ./tests
EIGEN = /usr/local/include/eigen3/

# GTest
LIB = /usr/local/lib
GTEST = $(LIB)/libgtest.a $(LIB)/libgtest_main.a -lpthread

# File pairs

# all
all: test

# default
default: neural_network

# neural_network
neural_network: $(SRC)/NeuralNetwork.cpp $(SRC)/NeuralNetwork.hpp
	$(CC) $(CFLAGS) -I $(EIGEN) $(SRC)/NeuralNetwork.cpp -o $(BIN)/neural_network

# Tests
test: $(TESTS)/test_NeuralNetwork.cpp
	$(CC) $(CFLAGS) -I $(EIGEN) $(GTEST) $(SRC)/NeuralNetwork.cpp $(TESTS)/test_NeuralNetwork.cpp -o $(BIN)/test

# Clean
clean:
	rm $(BIN)/*