# Current make system
BIN=./bin
SOURCE=./src

main.o: $(SOURCE)/main.cpp
	g++ $(SOURCE)/main.cpp -o $(BIN)/main.o