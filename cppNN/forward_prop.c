#include "forward_prop.h"
#include "activation.h"

void forward_propagation(NeuralNetwork *nn, double *input, double *hidden, double *output) {
    // Hidden layer
    for (int i = 0; i < nn->hidden_size; i++) {
        hidden[i] = 0;
        for (int j = 0; j < nn->input_size; j++) {
            hidden[i] += input[j] * nn->weights1[i * nn->input_size + j];
        }
        hidden[i] = sigmoid(hidden[i] + nn->bias1[i]);
    }

    // Output layer
    for (int i = 0; i < nn->output_size; i++) {
        output[i] = 0;
        for (int j = 0; j < nn->hidden_size; j++) {
            output[i] += hidden[j] * nn->weights2[i * nn->hidden_size + j];
        }
        output[i] = sigmoid(output[i] + nn->bias2[i]);
    }
}