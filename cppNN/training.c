#include "training.h"
#include "forward_prop.h"
#include "activation.h"

void train(NeuralNetwork *nn, double *input, double *target, int epochs, double learning_rate) {
    double hidden[nn->hidden_size];
    double output[nn->output_size];

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward propagation
        forward_propagation(nn, input, hidden, output);

       
        // Output layer
        double delta_output[nn->output_size];
        for (int i = 0; i < nn->output_size; i++) {
            delta_output[i] = (target[i] - output[i]) * sigmoid_derivative(output[i]);
        }

        // Hidden layer
        double delta_hidden[nn->hidden_size];
        for (int i = 0; i < nn->hidden_size; i++) {
            delta_hidden[i] = 0;
            for (int j = 0; j < nn->output_size; j++) {
                delta_hidden[i] += delta_output[j] * nn->weights2[j * nn->hidden_size + i];
            }
            delta_hidden[i] *= sigmoid_derivative(hidden[i]);
        }

        // Update weights and biases
        // Output layer
        for (int i = 0; i < nn->output_size; i++) {
            for (int j = 0; j < nn->hidden_size; j++) {
                nn->weights2[i * nn->hidden_size + j] += learning_rate * delta_output[i] * hidden[j];
            }
            nn->bias2[i] += learning_rate * delta_output[i];
        }

        // Hidden layer
        for (int i = 0; i < nn->hidden_size; i++) {
            for (int j = 0; j < nn->input_size; j++) {
                nn->weights1[i * nn->input_size + j] += learning_rate * delta_hidden[i] * input[j];
            }
            nn->bias1[i] += learning_rate * delta_hidden[i];
        }
    }
}