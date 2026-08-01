# hpc-classifier

An MNIST handwritten-digit classifier written from scratch in C, built two ways: a single-process trainer and an MPI-distributed version using a parameter-server pattern across multiple processes.

The neural network itself (forward pass, backprop, weight updates) is implemented directly with raw arrays and loops — no ML framework — so the actual math of a feedforward network is fully visible. The point of the assignment was the distributed-training side: taking that same network and training it in parallel across MPI ranks instead of a single process.

## Network

- Fully-connected feedforward network: 784 inputs (28×28 MNIST pixels) → 64 hidden units → 10 outputs (digit classes)
- ReLU activation on the hidden layer, softmax on the output layer
- Manual backpropagation and SGD weight updates (learning rate 0.01), no autograd
- Trains for up to 40 epochs, stopping early if test accuracy stops improving between epochs

The core network/training code (`neural.c`/`neural.h`) is based on [dlidstrom/NeuralNetworkInAllLangs](https://github.com/dlidstrom/NeuralNetworkInAllLangs), adapted for the MNIST problem specifically (ReLU/softmax activations, MNIST-shaped I/O).

## Distributed training (MPI)

`classifier_mpi` runs the same network across multiple MPI processes using a parameter-server layout:

- **Rank 0** acts as the parameter server: it holds the canonical weights, broadcasts them to workers each epoch, and reduces (sums) their updated weights back after training.
- **Ranks 1+** are workers: each loads its own shard of the training set (`mpi_load_images`/`mpi_load_labels` split the MNIST file by rank so no worker reads data another rank already owns), trains locally on its shard for the epoch, and sends its updated weights back to rank 0 via `MPI_Reduce`.
- Every epoch is a broadcast → local train → reduce cycle, keeping all ranks' networks in sync at each epoch boundary.

This is the same underlying network as the serial version — the difference is purely in how training data and weight updates are partitioned and synchronized across processes.

## Files

| File | Description |
|------|-------------|
| `neural.c` / `neural.h` | Network definition, forward pass, backprop, training step |
| `loader.c` / `loader.h` | MNIST IDX file parsing (images + labels), including per-rank data sharding for MPI |
| `main.c` | Single-process training/testing entry point → builds `classifier` |
| `mpi_main.c` | MPI parameter-server entry point → builds `classifier_mpi` |
| `Makefile` | Builds both binaries (`gcc` for the serial version, `mpicc` for the MPI version) |
| `run.sh` | Builds and runs the serial classifier |
| `mpi_run.sh` | Builds and runs the MPI classifier across 4 processes |

## Requirements

- GCC
- An MPI implementation providing `mpicc` and `mpiexec` (e.g. OpenMPI or MPICH), for the distributed version
- The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) in IDX format, placed under `datasets/`:
  - `datasets/train-images.idx3-ubyte`
  - `datasets/train-labels.idx1-ubyte`
  - `datasets/test-images.idx3-ubyte`
  - `datasets/test-labels.idx1-ubyte`

## Building

```sh
make        # builds both classifier and classifier_mpi
make clean
```

## Running

Serial version:

```sh
./run.sh
# or directly:
./classifier
```

MPI version (4 processes: 1 parameter server + 3 workers):

```sh
./mpi_run.sh
# or directly:
mpiexec -n 4 ./classifier_mpi
```

Both print per-epoch training accuracy to stdout; the serial version also reports final test-set accuracy once training completes.
