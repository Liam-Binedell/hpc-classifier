#ifndef LOADER_H_
#define LOADER_H_
#include <stdint.h>
#include <stdio.h>

#define IMAGE_SIZE 784

typedef struct {
    float **data;
    uint32_t size;
} Images;

typedef struct {
    uint8_t *data;
    uint32_t size;
} Labels;

uint32_t read_uint32(FILE *f);
void load_images(const char *path, Images *imgs);
void load_labels(const char *path, Labels *lbls);
void free_images(Images *imgs);

#endif // LOADER_H_
