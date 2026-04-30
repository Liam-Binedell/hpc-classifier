#include "neural.h"
#include <math.h>
#include <stdlib.h>
/*
 Refer to header file for proper accreditation of source code
*/

double relu(double z) { return z > 0 ? z : 0; }
double softmax(double z, double *z_i, int size) {
    double max = z_i[0];
    for (int i = 1; i < size; i++)
        if (z_i[i] > max)
            max = z_i[i];

    double exp_sum = 0;
    for (int i = 0; i < size; i++)
        exp_sum += exp(z_i[i] - max);

    return exp(z - max) / exp_sum;
}

double rand_weight() { return ((double)rand() / RAND_MAX) * 2.0 - 1.0; }

void network_init(Network *network, uint32_t n_inputs, uint32_t n_hidden, uint32_t n_outputs) {
    network->n_inputs = n_inputs;
    network->n_hidden = n_hidden;
    network->n_outputs = n_outputs;

    /*
      Github source uses calloc in every instance.
      I have elected to use malloc in cases where zero initialisation is
      not a priority, as it has less overhead and is quicker.
    */
    network->weights_hidden = (double *)malloc(n_inputs * n_hidden * sizeof(double));
    network->biases_hidden = (double *)calloc(n_hidden, sizeof(double));
    network->weights_output = (double *)malloc(n_hidden * n_outputs * sizeof(double));
    network->biases_output = (double *)calloc(n_outputs, sizeof(double));
    network->hidden = (double *)malloc(n_hidden * sizeof(double));
    network->output = (double *)malloc(n_outputs * sizeof(double));

    for (size_t i = 0; i < n_inputs * n_hidden; i++) {
        network->weights_hidden[i] = rand_weight();
    }

    for (size_t i = 0; i < n_hidden * n_outputs; i++) {
        network->weights_output[i] = rand_weight();
    }
}

void network_free(Network *network) {
    free(network->weights_hidden);
    free(network->biases_hidden);
    free(network->weights_output);
    free(network->biases_output);
    free(network->hidden);
    free(network->output);
}

void network_predict(Network *network, const float *input) {
    for (uint32_t c = 0; c < network->n_hidden; c++) {
        double sum = 0;
        for (uint32_t r = 0; r < network->n_inputs; r++)
            sum += input[r] * network->weights_hidden[r * network->n_hidden + c];
        network->hidden[c] = relu(sum + network->biases_hidden[c]);
    }

    double temp[network->n_outputs];
    for (uint32_t c = 0; c < network->n_outputs; c++) {
        double sum = 0;
        for (uint32_t r = 0; r < network->n_hidden; r++)
            sum += network->hidden[r] * network->weights_output[r * network->n_outputs + c];
        temp[c] = sum + network->biases_output[c];
    }

    for (uint32_t c = 0; c < network->n_outputs; c++)
        network->output[c] = softmax(temp[c], temp, network->n_outputs);
}

void trainer_init(Trainer *trainer, Network *network) {
    trainer->grad_hidden =
        calloc(network->n_hidden, sizeof(*trainer->grad_hidden));
    trainer->grad_output =
        calloc(network->n_outputs, sizeof(*trainer->grad_output));
}

void trainer_train(Trainer *trainer, Network *network, const float *input, const double *y, double lr) {
    network_predict(network, input);
    for (uint32_t c = 0; c < network->n_outputs; c++) {
        trainer->grad_output[c] = network->output[c] - y[c];
    }
    for (uint32_t r = 0; r < network->n_hidden; r++) {
        double sum = 0.0;
        for (uint32_t c = 0; c < network->n_outputs; c++) {
            sum += trainer->grad_output[c] *
                   network->weights_output[r * network->n_outputs + c];
        }
        trainer->grad_hidden[r] = sum * (network->hidden[r] > 0 ? 1.0 : 0.0);
    }
    for (size_t r = 0; r < network->n_hidden; r++) {
        for (size_t c = 0; c < network->n_outputs; c++) {
            network->weights_output[r * network->n_outputs + c] -=
                lr * trainer->grad_output[c] * network->hidden[r];
        }
    }
    for (size_t r = 0; r < network->n_inputs; r++) {
        for (size_t c = 0; c < network->n_hidden; c++) {
            network->weights_hidden[r * network->n_hidden + c] -=
                lr * trainer->grad_hidden[c] * input[r];
        }
    }
    for (size_t c = 0; c < network->n_outputs; c++) {
        network->biases_output[c] -= lr * trainer->grad_output[c];
    }
    for (size_t c = 0; c < network->n_hidden; c++) {
        network->biases_hidden[c] -= lr * trainer->grad_hidden[c];
    }
}

void trainer_free(Trainer *trainer) {
    free(trainer->grad_hidden);
    free(trainer->grad_output);
}
