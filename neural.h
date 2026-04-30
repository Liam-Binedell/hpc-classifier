#ifndef NEURAL_H_
#define NEURAL_H_

/*
  CREDIT: https://github.com/dlidstrom/NeuralNetworkInAllLangs
  Modified (ever so slightly) to better suit MNIST classification problem by me :)
*/

#include <stdint.h>

// My chosen activation funcs to help with MNIST problem
double relu(double z);
double softmax(double z, double *z_i, int size);

double rand_weight();

typedef struct {
    double *weights_hidden;
    double *biases_hidden;
    double *weights_output;
    double *biases_output;
    double *hidden;
    double *output;
    uint32_t n_inputs;
    uint32_t n_hidden;
    uint32_t n_outputs;
} Network;

void network_init(Network *network, uint32_t n_inputs, uint32_t n_hidden, uint32_t n_outputs);
void network_free(Network *network);
void network_predict(Network *network, const float *input);

typedef struct {
    double *grad_hidden;
    double *grad_output;
} Trainer;

void trainer_init(Trainer *trainer, Network *network);
void trainer_train(Trainer *trainer, Network *network, const float *input, const double *output, double lr);
void trainer_free(Trainer *trainer);

#endif // NEURAL_H_
