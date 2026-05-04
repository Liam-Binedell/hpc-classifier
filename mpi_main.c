#include "loader.h"
#include "neural.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define EPOCHS 40
#define LR 0.01
#define IMAGE_SIZE 784
#define N_OUTPUTS 10

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
    int processes, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    Images train_imgs = {NULL, 0, 0, 0, 0};
    Labels train_lbls = {NULL, 0};

    Images test_imgs = {NULL, 0, 0, 0, 0};
    Labels test_lbls = {NULL, 0};

    Network network;
    Trainer trainer;

    network_init(&network, IMAGE_SIZE, 64, N_OUTPUTS);


    printf("poes\n");
    return 0;
}
