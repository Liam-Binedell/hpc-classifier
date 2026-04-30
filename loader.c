#include "loader.h"
#include <byteswap.h>
#define ERR(format, ...)                             \
    {                                                \
        fprintf(stderr, format "\n", ##__VA_ARGS__); \
        exit(1);                                     \
    }
// This converts the mnist headers from big endian to little endian
// with the method provided by byteswap.h
uint32_t read_uint32(FILE *f) {
    uint32_t val;
    fread(&val, sizeof(val), 1, f);
    return bswap_32(val);
}

void load_images(const char *path, Images *imgs) {
    FILE *f = fopen(path, "rb");
    if (!f)
        ERR("Could not open training images file");

    read_uint32(f); // parse and skip magic number
    // Parsing the mnist headers
    imgs->size = read_uint32(f);
    imgs->n_rows = read_uint32(f);
    imgs->n_cols = read_uint32(f);

    imgs->img_size = imgs->n_rows * imgs->n_cols;
    printf("Size of each image: %u\n", imgs->img_size);
    imgs->data = (float **)malloc(imgs->size * sizeof(float *));
    if (!imgs->data)
        ERR("Could not allocate memory for images");

    for (int i = 0; i < imgs->size; i++) {
        imgs->data[i] = (float *)malloc(imgs->img_size * sizeof(float));
        if (!imgs->data[i])
            ERR("Could not allocate memory for image %d", i);

        for (int j = 0; j < imgs->img_size; j++) {
            uint8_t pixel;
            fread(&pixel, 1, 1, f);
            imgs->data[i][j] = pixel/255.0f; // division to normalise values in range [0,1]
        }
    }
    fclose(f);
}

void load_labels(const char *path, Labels *lbls) {
    FILE *f = fopen(path, "rb");
    if (!f)
        ERR("Could not open training labels file");

    read_uint32(f); // parse and skip magic number
    // Parsing mnist headers
    lbls->size = read_uint32(f);

    lbls->data = (uint8_t *)malloc(lbls->size * sizeof(uint8_t));
    if (!lbls->data)
        ERR("Could not allocate memory for labels");

    fread(lbls->data, 1, lbls->size, f);

    fclose(f);
}

void free_images(Images *imgs) {
    for (int i = 0; i < imgs->size; i++) {
        free(imgs->data[i]);
    }
    free(imgs->data);
}
