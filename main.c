#include "loader.h"
#include "neural.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EPOCHS 40

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

    Images train_imgs = {NULL, 0, 0, 0, 0};
    Labels train_lbls = {NULL, 0};

    Images test_imgs = {NULL, 0, 0, 0, 0};
    Labels test_lbls = {NULL, 0};
    Network network;
    Trainer trainer;

    load_images("datasets/train-images.idx3-ubyte", &train_imgs);
    load_labels("datasets/train-labels.idx1-ubyte", &train_lbls);
    printf("Loaded %u images\n", train_imgs.size);
    printf("Loaded %u labels\n", train_lbls.size);

    load_images("datasets/test-images.idx3-ubyte", &test_imgs);
    load_labels("datasets/test-labels.idx1-ubyte", &test_lbls);
    printf("Loaded %u test images\n", test_imgs.size);
    printf("Loaded %u test labels\n", test_lbls.size);

    network_init(&network, IMAGE_SIZE, 64, N_OUTPUTS);
    trainer_init(&trainer, &network);

    int correct;
    double accuracy, prev_accuracy, train_accuracy;
    double y[N_OUTPUTS];

    // TRAIN FOR THE SET AMOUNT OF EPOCHS
    // OR UNTIL A SATISFACTORY LEVEL OF CONVERGENCE IS ACHIEVED
    accuracy = 0.0;
    prev_accuracy = 0.0;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        correct = 0;
        for (int i = 0; i < train_imgs.size; i++) {
            one_hot(train_lbls.data[i], y);
            trainer_train(&trainer, &network, train_imgs.data[i], y, LR);
            if (argmax(network.output, N_OUTPUTS) == train_lbls.data[i])
                correct++;
        }
        prev_accuracy = accuracy;
        train_accuracy = (double)correct / train_imgs.size * 100.0;

        correct = 0;
        for (int i = 0; i < test_imgs.size; i++) {
            network_predict(&network, test_imgs.data[i]);
            if (argmax(network.output, N_OUTPUTS) == test_lbls.data[i])
                correct++;
        }
        accuracy = (double)correct / test_imgs.size * 100.0;
        if (accuracy - prev_accuracy < 0.0)
            break;


        printf("Epoch %d — Accuracy: %.2f%%\n", epoch + 1, train_accuracy);
    }

    // TEST TRAINED MODEL ON UNSEEN DATA
    correct = 0;
    for (int i = 0; i < test_imgs.size; i++) {
        network_predict(&network, test_imgs.data[i]);
        if (argmax(network.output, N_OUTPUTS) == test_lbls.data[i])
            correct++;
    }
    printf("FINAL TESTING ACCURACY: %.2f%%\n",
           (double)correct / test_imgs.size * 100.0);

    trainer_free(&trainer);
    network_free(&network);
    free_images(&train_imgs);
    free(train_lbls.data);
    free_images(&test_imgs);
    free(test_lbls.data);
    return 0;
}
