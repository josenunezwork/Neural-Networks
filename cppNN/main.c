#include <stdio.h>
#include "nn_structure.h"
#include "forward_prop.h"
#include "training.h"

int main() {
    NeuralNetwork *nn = create_neural_network(2, 2, 1);
    
    double input[2] = {0.5, 0.1};
    double target[1] = {0.7};
    
    train(nn, input, target, 1000, 0.1);
    
    double hidden[2], output[1];
    forward_propagation(nn, input, hidden, output);
    
    printf("Output: %f\n", output[0]);
    
    free_neural_network(nn);
    return 0;
}