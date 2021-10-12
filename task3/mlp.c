#include <time.h>
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
        layer->W[w] = new_variable(rand() / (double) RAND_MAX * 0.01);
    }
    return layer;
}

Double** layer(Layer* layer, Double** X) {
    Double** outputs = (Double**) malloc(sizeof(Double*) * layer->n_outputs);
    for(int i = 0; i < layer->n_outputs; i++) {
        outputs[i] = new_constant(0.0);
        for(int j = 0; j < layer->n_inputs; j++) {
            outputs[i] = add(outputs[i], multiply(X[j], layer->W[i * layer->n_inputs + j]));
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
    Double* sum = new_constant(0.0);
    for(int i = 0; i < n; i++) {
        Double* err = subtract(gt[i], pred[i]);
        sum = add(sum, multiply(err, err));
    }
    return divide(sum, new_constant(n));
}

int main(int n_args, char** args) {
    srand(time(0));

    /* read parameters */
    if (n_args != 4) {
        printf("Required parameters: train_dataset_filename, n_epochs, learning_rate\n");
        exit(1);
    }
    char* train_ds_filename = args[1];
    int n_epochs = atoi(args[2]);
    double learning_rate = atof(args[3]);

    printf("Reading dataset... ");
    FILE* train_ds_file = fopen(train_ds_filename, "r");
    int n_samples, n_inputs, n_outputs;
    fscanf(train_ds_file, "%i %i %i", &n_samples, &n_inputs, &n_outputs);
    printf("n_samples: %i, n_inputs: %i, n_outputs: %i\n", n_samples, n_inputs, n_outputs);
    Double** X = (Double**) malloc(sizeof(Double*) * n_samples * n_inputs);
    Double** Y = (Double**) malloc(sizeof(Double*) * n_samples * n_outputs);
    float tmp;
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_inputs; j++) {
            fscanf(train_ds_file, "%f", &tmp);
            X[i * n_inputs + j] = new_variable(tmp);
        }
        for (int j = 0; j < n_outputs; j++) {
            fscanf(train_ds_file, "%f", &tmp);
            Y[i * n_outputs + j] = new_variable(tmp);
        }
    }

    printf("Preparing model...\n");
    Layer* layers[2];
    layers[0] = create_layer(4, 2, 1);
    layers[1] = create_layer(2, 4, 0);

    printf("Training...\n");
    for (int e = 0; e < n_epochs; e++) {
        for (int s = 0; s < n_samples - 1; s++) {
            Double** pred = forward_mlp(layers, 2, X + (s * n_inputs));
            Double* mse = loss(pred, Y + (s * n_outputs), n_outputs);
            printf("%f\n", mse->value);
            backpropagate(mse);

            for(int i = 0; i < 2; i++) {
                for(int j = 0; j < layers[i]->n_outputs; j++) {
                    layers[i]->W[j]->value -= layers[i]->W[j]->derivative * learning_rate;
                    layers[i]->W[j]->derivative = 0.0;
                }
            }
        }
    }

    printf("Test...\n"); /* hardcoded for 4 inputs and 4 outputs */
    for (int s = 0; s < n_samples; s++) {
        Double** x = X + (s * n_inputs);
        Double** result = forward_mlp(layers, 2, x);
        printf("UWAGA: %f %f %f %f   --->   %f %f %f %f\n", 
                x[0]->value, x[1]->value, x[2]->value, x[3]->value,
                result[0]->value, result[1]->value, result[2]->value, result[3]->value);
    }
}
