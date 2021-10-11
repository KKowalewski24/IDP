#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/* This is double wrapper, which remembers args and local gradient of args of
 * the function, which is the source of it. It also can store derivative of
 * this variable, via backpropagation. */
typedef struct Double {
    double value;
    double derivative;
    int n_args;
    struct Double** args;
    double* local_gradient;
} Double;

static Double* one_arg_function(Double* arg1, double derivative, double value) {
    Double* d = (Double*) malloc(sizeof(Double));
    d->value = value;
    d->derivative = 0.0;
    d->n_args = 1;
    d->args = (Double**) malloc(sizeof(Double*));
    d->args[0] = arg1;
    d->local_gradient = (double*) malloc(sizeof(double));
    d->local_gradient[0] = derivative;
    return d;
}

static Double* two_args_function(Double* arg1, Double* arg2, double derivative1, double derivative2, double value) {
    Double* d = (Double*) malloc(sizeof(Double));
    d->value = value;
    d->derivative = 0.0;
    d->n_args = 2;
    d->args = (Double**) malloc(sizeof(Double*) * 2);
    d->args[0] = arg1;
    d->args[1] = arg2;
    d->local_gradient = (double*) malloc(sizeof(double) * 2);
    d->local_gradient[0] = derivative1;
    d->local_gradient[1] = derivative2;
    return d;
}

Double* new_double(double value) {
    Double* d = (Double*) malloc(sizeof(Double));
    d->value = value;
    d->derivative = 0.0;
    d->n_args = 0;
    d->args = NULL;
    d->local_gradient = NULL;
    return d;
}

Double* add(Double* a, Double* b) {
    return two_args_function(a, b, 1.0, 1.0, a->value + b->value);
}

Double* subtract(Double* a, Double* b) {
    return two_args_function(a, b, 1.0, -1.0, a->value - b->value);
}

Double* multiply(Double* a, Double* b) {
    return two_args_function(a, b, b->value, a->value, a->value * b->value);
}

Double* divide(Double* a, Double* b) {
    return two_args_function(a, b, 1.0 / b->value, -a->value / (b->value * b->value), a->value / b->value);
}

Double* power(Double* a, Double* b) {
    return two_args_function(a, b, 
            b->value * pow(a->value, b->value - 1.0), 
            pow(a->value, b->value) * log(a->value), 
            pow(a->value, b->value));
}

Double* exponent(Double* a) {
    return one_arg_function(a, exp(a->value), exp(a->value));
}

Double* logarithm(Double* a) {
    return one_arg_function(a, 1 / a->value, log(a->value));
}

static void recurrent_backpropagate(Double* result) {
    for(int i = 0; i < result->n_args; i++) {
        result->args[i]->derivative += result->derivative * result->local_gradient[i];
        recurrent_backpropagate(result->args[i]);
    }
    if (result->n_args > 0) {
        free(result->local_gradient);
        free(result->args);
        free(result);
    }
}

void backpropagate(Double* result) {
    result->derivative = 1.0;
    recurrent_backpropagate(result);
}

//////////////////////////////////////////////////////////////////////
// Double* test(Double* x, Double* y, Double* z) {
//     return add(multiply(multiply(x, x), y), add(y, z));
// }
// 
// int main() {
//     Double* x = new_double(3.0);
//     Double* y = new_double(4.0);
//     Double* z = new_double(2.0);
//     backpropagate(test(x, y, z));
//     printf("%f %f\n", x->derivative, y->derivative);
// }
//////////////////////////////////////////////////////////////////////
