# Current make system
BIN = ./bin
SOURCE = ./src
GTEST = ./googletest/googletest/src
TEST = ./tests
CFLAGS = -std=c++11 -stdlib=libc++
GTESTFLAGS = -lgtest -lpthread


test_NeuralNetwork.o: $(SOURCE)/main.cpp
	g++ $(CLAGS) $(SOURCE)/NeuralNetwork.cpp $(TEST)/test_NeuralNetwork.cpp $(GTEST)/gtest_main.cc $(GTESTFLAGS) -o $(BIN)/test_NeuralNetwork.o