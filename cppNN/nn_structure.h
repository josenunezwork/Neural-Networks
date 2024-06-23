#ifndef NN_STRUCTURE_H
#define NN_STRUCTURE_H

#include <stdlib.h>

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double *weights1;
    double *weights2;
    double *bias1;
    double *bias2;
} NeuralNetwork;

NeuralNetwork* create_neural_network(int input_size, int hidden_size, int output_size);
void free_neural_network(NeuralNetwork *nn);

#endif