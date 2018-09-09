# ANN
This is my own implementation for a feed-forward artificial neural network (ANN).  I have not written in C++ in quite a bit, so pardon any distractions.

## Requirements

I compiled this using a Mac, and g++ (or really clang) version 4.2.1. Other libraries required to run include:

* Eigen: http://eigen.tuxfamily.org/index.php?title=Main\_Page
* GoogleTest (link to my forked repo): https://github.com/dadler6/googletest 

## Compiling instructions.

If all requirements are satisfied, you can compile using the makefile.

**Compiling**

To compile the Neural Network for running, type

```make run```

This will create a run executable in bin/run.

To compile tests, type

```make test_neural_network```

which will make a test executable in bin/test_neural_network, to test NeuralNetwork.hpp/cpp.

or

```make test_dataio```

which will make a test executable in bin/test_dataio, to test DataIO.hpp/cpp.

**Running**

To run the test code (test_neural_network or test_dataio), run either of the following two lines after compiling:

```
    ./bin/test_neural_network
    ./bin/test_dataio
```

To train data, after compiling the run file, you will need a training dataset, saved in a .csv file, where the last column is the target values, and the first row are column headers, and a configuration file that specifies how many nodes should be within each network layer.  You can then execute training by typing:

```
    ./bin/run train path/to/train/data.csv output/file/path learning_rate threshold number_of_layers path/to/network/config/file.csv
```

An example of training data for the "XOR" case can be found in data/test_data_1.csv and data/test_config_1.csv.  This code was run with the following command:

```
    ./bin/run train data/test_data_1.csv data/test_network_1.txt 0.1 0.731 3 data/test_config_1.csv
```

This will save the trained network in a file called data/test_network_1.txt.  To then predict new data, you will need a saved .csv file with the matrix to predict target values on, saved with the first row as the column headers.  You can then run the following code:

```
    ./bin/run predict path/to/data.csv output/file/path.txt path/to/trained/network
```

This will save the predicted target values to the designated output path.  For instance, within the above "XOR" training data executable call, I can predict values after taining (using the example data in data/test_predict_1.csv) by typing:

```
    ./bin/run predict data/test_predict_1.csv data/test_ans_1.txt data/test_network_1.txt
```

Which will saved the predicted target values in a file called data/test_ans_1.txt

## Folder structure

**src**

This is the source code for the ANN implementation.  Files include:

* NeuralNetwork.hpp: The header file with declarations for the neuralnetwork namespace and NeuralNetwork class.
* NeuralNetwork.cpp: The definitions for the methods of the NeuralNetwork class.
* DataIO.hpp: The header file of declarations to read in/read out data to train/predict.
* DataIO.cpp: The definitions for the methods within DataIO.hpp
* main.cpp: Runtime file for either training or predicting given data.

**bin**

Binaries for the neural network implementation.  These are not included within the GitHub repo online, as one should compile the files locally after cloning the repo, and compile the src files according to one's computer specifications.

**tests**

Test cases for the code within the src folder.  Tests include:

* test_NeuralNetwork.cpp: Basic tests showing how the NeuralNetwork object defined in NeuralNetwork.cpp can train and predict boolean operators.
* test_DataIO.cpp: Basic tests showing functionality of input/output code for training/prediction.

**data**

Data I've used to show functionality of the network. Files include:

* test_data_1.csv: Example training data for "XOR" logic.  Last column are target values.
* test_config_1.csv: Configuration file for the "XOR" network.
* test_network_1.txt: The trained network for the "XOR" logic.
* test_predict_1.txt: The "XOR" matrix again without target labels, used for testing prediction after training.
* test_ans_1.txt: The predicted target values for the "XOR" logic.

**examples**

Examples on how to utilize the ANN created.  This will be later populated with Python notebooks.