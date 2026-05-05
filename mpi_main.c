#include "loader.h"
#include "neural.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define EPOCHS 40

void broadcast_network(Network *network) {
    MPI_Bcast(network->weights_hidden, network->n_inputs * network->n_hidden, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(network->biases_hidden, network->n_hidden, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(network->weights_output, network->n_hidden * network->n_outputs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(network->biases_output, network->n_outputs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void reduce_network(Network *network, Network *global) {
    MPI_Reduce(network->weights_hidden, global->weights_hidden, network->n_inputs * network->n_hidden, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(network->biases_hidden, global->biases_hidden, network->n_hidden, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(network->weights_output, global->weights_output, network->n_hidden * network->n_outputs, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(network->biases_output, global->biases_output, network->n_outputs, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

void one_hot(uint8_t label, double *vec) {
    for (int i = 0; i < N_OUTPUTS; i++)
        vec[i] = 0.0;
    vec[label] = 1.0;
}

int argmax(double *vec, int size) {
    int max_i = 0;
    for (int i = 1; i < size; i++)
        if (vec[i] > vec[max_i])
            max_i = i;
    return max_i;
}

int main(int argc, char **argv) {
    srand(time(NULL));
    MPI_Init(&argc, &argv);
    int n_workers, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_workers);

    Images train_imgs = {NULL, 0};
    Labels train_lbls = {NULL, 0};

    Images test_imgs = {NULL, 0};
    Labels test_lbls = {NULL, 0};

    Network network, global;
    Trainer trainer;

    network_init(&global, IMAGE_SIZE, 64, N_OUTPUTS);
    network_init(&network, IMAGE_SIZE, 64, N_OUTPUTS);
    trainer_init(&trainer, &network);

    broadcast_network(&network);

    if (rank != 0) {
        mpi_load_images("datasets/train-images.idx3-ubyte", &train_imgs, rank, n_workers);
        mpi_load_labels("datasets/train-labels.idx1-ubyte", &train_lbls, rank, n_workers);
        printf("Process %d loaded %u images\n", rank, train_imgs.size);
        printf("Process %d loaded %u labels\n", rank, train_lbls.size);
    }

    int correct;
    double accuracy, prev_accuracy;
    double y[N_OUTPUTS];

    prev_accuracy = accuracy = 0;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        if (rank != 0) {
            correct = 0;
            for (int i = 0; i < train_imgs.size; i++) {
                one_hot(train_lbls.data[i], y);
                trainer_train(&trainer, &network, train_imgs.data[i], y, LR);
                if (argmax(network.output, N_OUTPUTS) == train_lbls.data[i])
                    correct++;
            }
        }
        reduce_network(&network, &global);
        if (rank == 0) {
            network = global;
        }
        broadcast_network(&network);
    }

    if (rank != 0) {
        free_images(&train_imgs);
        free(train_lbls.data);
    }
    network_free(&network);
    network_free(&global);
    trainer_free(&trainer);
    MPI_Finalize();
    return 0;
}
