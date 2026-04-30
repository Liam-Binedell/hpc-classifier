#include "loader.h"
#include "neural.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EPOCHS 10
#define LR 0.01
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

int main() {
    srand(time(NULL));

    Images imgs = {NULL, 0, 0, 0, 0};
    Labels lbls = {NULL, 0};
    Network network;
    Trainer trainer;

    load_images("datasets/train-images.idx3-ubyte", &imgs);
    load_labels("datasets/train-labels.idx1-ubyte", &lbls);
    printf("Loaded %u images\n", imgs.size);
    printf("Loaded %u labels\n", lbls.size);

    network_init(&network, imgs.img_size, 64, N_OUTPUTS);
    trainer_init(&trainer, &network);

    double y[N_OUTPUTS];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        int correct = 0;
        for (int i = 0; i < imgs.size; i++) {
            one_hot(lbls.data[i], y);
            trainer_train(&trainer, &network, imgs.data[i], y, LR);
            if (argmax(network.output, N_OUTPUTS) == lbls.data[i])
                correct++;
        }
        printf("Epoch %d — Accuracy: %.2f%%\n", epoch + 1,
               (double)correct / imgs.size * 100.0);
    }

    trainer_free(&trainer);
    network_free(&network);
    free_images(&imgs);
    free(lbls.data);
    return 0;
}
