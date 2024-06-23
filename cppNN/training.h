#ifndef TRAINING_H
#define TRAINING_H

#include "nn_structure.h"

void train(NeuralNetwork *nn, double *input, double *target, int epochs, double learning_rate);

#endif