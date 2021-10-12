#include <stdlib.h>
#include <stdio.h>
#include "autograd.h"

typedef struct Layer {
    int n_inputs;
    int n_outputs;
    int sigmoid;
    Double** W;
} Layer;

Layer* create_layer(int n_inputs, int n_outputs, int sigmoid) {
    Layer* layer = (Layer*) malloc(sizeof(Layer));
    layer->n_inputs = n_inputs;
    layer->n_outputs = n_outputs;
    layer->sigmoid = sigmoid;
    layer->W = (Double**) malloc(sizeof(Double*) * n_inputs * n_outputs);
    for (int w = 0; w < n_inputs * n_outputs; w++) {
        layer->W[w] = new_double(rand() / (double) RAND_MAX * 0.1);
    }
    return layer;
}

Double** layer(Layer* layer, Double** X) {
    Double** outputs = (Double**) malloc(sizeof(Double*) * layer->n_outputs);
    for(int i = 0; i < layer->n_outputs; i++) {
        for(int j = 0; j < layer->n_inputs; j++) {
            if (j == 0) {
                outputs[i] = multiply(X[j], layer->W[i * layer->n_inputs + j]);
            } else {
                outputs[i] = add(outputs[i], multiply(X[j], layer->W[i * layer->n_inputs + j]));
            }
        }
        if (layer->sigmoid) {
            outputs[i] = sigmoid(outputs[i]);
        }
    }
    return outputs;
}

Double** forward_mlp(Layer** layers, int n_layers, Double** X) {
    for(int i = 0; i < n_layers; i++) {
        X = layer(layers[i], X);
    }
    return X;
}

Double* loss(Double** pred, Double** gt, int n) {
    Double* result;
    for(int i = 0; i < n; i++) {
        Double* err = subtract(gt[i], pred[i]);
        if (i == 0) {
            result = multiply(err, err);
        } else {
            result = add(result, multiply(err, err));
        }
    }
    return result;
}

int main() {
    Layer* layers[2];
    layers[0] = create_layer(4, 2, 1);
    layers[1] = create_layer(2, 4, 1);

    Double** X = (Double**) malloc(sizeof(Double*) * 4);
    X[0] = new_double(1.0);
    X[1] = new_double(0.0);
    X[2] = new_double(0.0);
    X[3] = new_double(0.0);

    for(int e = 0; e < 1000; e++) {
        Double** pred = forward_mlp(layers, 2, X);
        Double* mse = loss(pred, X, 4);
        printf("%f\n", mse->value);
        backpropagate(mse);

        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < layers[i]->n_outputs; j++) {
                layers[i]->W[j]->value -= layers[i]->W[j]->derivative * 0.1;
                layers[i]->W[j]->derivative = 0.0;
            }
        }
    }

    Double** result = forward_mlp(layers, 2, X);
    printf("UWAGA: %f %f %f %f", result[0]->value, result[1]->value, result[2]->value, result[3]->value);
}
