#include "activation.h"
#include <math.h>

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1 - x);
}