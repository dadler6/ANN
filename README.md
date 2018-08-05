# ANN
This is my own implementation for a feed-forward artificial neural network (ANN).  I have not written C++ in quite a bit, so pardon any distractions.


## Requirements

I compiled this using a Mac, and gpp (or really clang) version 4.2.1. Other libraries required to run include:

* Eigen: http://eigen.tuxfamily.org/index.php?title=Main\_Page
* GoogleTest (link to my forked repo): https://github.com/dadler6/googletest 

## Compiling instructions.

If all requirements are satisfied, you can combile using the make file.  I will list how to compile using the makefile, and any specific instructions for test cases, etc as this project continues.


## Folder structure

**src**

This is the code for the ANN implementation.  Files include:

* NeuralNetwork.hpp: The header file with declarations for the neuralnetwork namespace and NeuralNetwork class.
* NeuralNetwork.cpp: The definitions for the methods of the NeuralNetwork class.
* STILL IN PROGRESS main.cpp: Runtime file for running either a training or prediction of a neural network given data.

**bin**

Binaries for the neural network implementation.  These are not included within the GitHub repo online, as one should compile the files locally after cloning the repo, and compile the src files according to one's computer specifications.

**tests**

Test cases for the code within the src folder.  Tests include:

* test_NeuralNetwork.cpp: Basic tests showing how the NeuralNetwork object defined in NeuralNetwork.cpp can train and predict boolean operators.
* STILL IN PROGRESS: test_main.cpp: Basic tests showing functionality of runtime code.

**data**

Data I use potentially within the test cases, exmaples.

**examples**

Examples on how to utilize the ANN created.

## Implementation Notes

There is a makefile included to compile if all libraries are installed on your computer.  The makefile currently supports the following commands:

* ```make test_neural_network```: Compiles the NeuralNetwork.cpp file and test_NeuralNetwork.cpp into a bin/test_neural_network binary.
