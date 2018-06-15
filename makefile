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

# all
all: test run

# default
default: run

# test
test: test

run: neural_network
	$(CC) $(CFLAGS) -I $(EIGEN) $(GTEST) $(SRC)/main.cpp $(BIN)/neural_network.o -o $(BIN)/run

# Tests
test: neural_network
	$(CC) $(CFLAGS) -I $(EIGEN) $(GTEST) $(TESTS)/test_NeuralNetwork.cpp $(BIN)/neural_network.o -o $(BIN)/test

# neural_network
neural_network:
	$(CC) -c $(CFLAGS) -I $(EIGEN) $(SRC)/NeuralNetwork.cpp -o $(BIN)/neural_network.o

# Clean
clean:
	rm $(BIN)/*