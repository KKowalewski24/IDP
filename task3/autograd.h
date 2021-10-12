#ifndef AUTOGRAD_H
#define AUTOGRAD_H

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

void delete_double(Double* d);

Double* new_variable(double value);

Double* new_constant(double value);

Double* add(Double* a, Double* b);

Double* subtract(Double* a, Double* b);

Double* multiply(Double* a, Double* b);

Double* divide(Double* a, Double* b);

Double* power(Double* a, Double* b);

Double* exponent(Double* a);

Double* logarithm(Double* a);

Double* sigmoid(Double* a);

void backpropagate(Double* result);

#endif //AUTOGRAD_H
