/**
 * DataIO.cpp
 * 
 * Written by Dan Adler
 * Email: daadler0309@gmail.com
 * GitHub: https://github.com/dadler6/
 * 
 * Implementation file to help with DataIO.  The file will:
 * 
 * 1) Build a neural network, through opening up a specified dataset
 *    (train and target), and fitting a neural network to this data,
 *    and then save the network to a specified file
 * 2) Predict on a network.  This will intake a training dataset,
 *    an already trained network, and then predict information
 *    from that network.
 * 
 * To train (1) you can use the following command:
 * ./run train input_data output_ann_file step_size threshold num_layers layers_config
 * 
 * To predict (2) you can use the following command:
 * ./run predict input_data output_data_file ann_file 
 * 
 * See DataIO.hpp for more definitions.
 */

// Include
#include "DataIO.hpp"


// Constants
size_t const VEC_SIZE = 100; 


MatrixXf open_data(string f) {
    // Read in file
    cout << "Reading in input data..." << endl;
    ifstream in;
    in.open(f);

    // Now make matrix, initialize vector to do so
    vector<vector<float> > cells;
    string line;
    
    // Count current lines
    int r = -1;
    int c;
    while(getline(in, line)) {
        stringstream lineStream(line);
        vector<float> *temp = new vector<float>(VEC_SIZE);
        string cell;
        c = 0;
        while((r >= 0) && getline(lineStream, cell, ',')) {
            (*temp)[c] = stod(cell);
            c++;
        }
        if (c > 0) {
            temp->resize(c);
            cells.push_back(*temp);
        }
        r++;
    }

    // Initialize matrix and return
    MatrixXf X_new(r, c);
    for (int i=0; i < (int) cells.size(); i++) {
        X_new.row(i) = VectorXf::Map(&cells[i][0], cells[i].size());
    }

    return X_new;
};


int train_network(
    MatrixXf X, 
    float eta, 
    float thresh,
    string output_filename,
    int num_layers,
    VectorXi config
    ) {
    // Get y data from X
    VectorXf y = X.rightCols(1);
    // Drop last column of X
    X = X.leftCols(X.cols() - 1);

    // Build a neural network
    NeuralNetwork *ann = new NeuralNetwork(num_layers, config, eta, thresh);

    // Fit the network
    cout << "Fitting network..." << endl;
    ann->fit(X, y);

    // Save network
    cout << "Saving network..." << endl;
    ofstream ofs(output_filename);
    ofs << ann;
    ofs.close();

    // End and delete
    cout << "Network saved!" << endl;
    delete ann;

    return 0;
};


int predict_values(MatrixXf X, string ann_filename, string output_filename) {

    // Load in file
    cout << "Reading in neural network..." << endl;
    ifstream ifs(ann_filename);
    NeuralNetwork *ann = new NeuralNetwork();
    ifs >> ann;

    // Predict data
    cout << "Predicting data..." << endl;
    VectorXf y = ann->predict(X);

    // End and delete
    cout << "Outputting data..." << endl;
    ofstream ofs(output_filename);
    if (ofs.is_open()) {
        ofs << "PREDICTED" << "\n" << y;
    }
    delete ann;

    return 0;
}
