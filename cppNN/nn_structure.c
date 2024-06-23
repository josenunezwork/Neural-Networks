#include "nn_structure.h"
#include <stdlib.h>
#include <time.h>

double random_weight() {
    return ((double)rand() / RAND_MAX) * 2 - 1;
}

NeuralNetwork* create_neural_network(int input_size, int hidden_size, int output_size) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    nn->weights1 = (double*)malloc(input_size * hidden_size * sizeof(double));
    nn->weights2 = (double*)malloc(hidden_size * output_size * sizeof(double));
    nn->bias1 = (double*)malloc(hidden_size * sizeof(double));
    nn->bias2 = (double*)malloc(output_size * sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < input_size * hidden_size; i++)
        nn->weights1[i] = random_weight();
    for (int i = 0; i < hidden_size * output_size; i++)
        nn->weights2[i] = random_weight();
    for (int i = 0; i < hidden_size; i++)
        nn->bias1[i] = random_weight();
    for (int i = 0; i < output_size; i++)
        nn->bias2[i] = random_weight();

    return nn;
}

void free_neural_network(NeuralNetwork *nn) {
    free(nn->weights1);
    free(nn->weights2);
    free(nn->bias1);
    free(nn->bias2);
    free(nn);
}