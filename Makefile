##
# Federated Learning
#
# @file
# @version 0.1

CC = gcc
MPICC= mpicc
CFLAGS = -Wall -O2
LDLIBS = -lm

TARGET1 = classifier
TARGET2 = classifier_mpi

# Common object files (library code)
COMMON_OBJS = loader.o neural.o

# Main objects (different for each binary)
MAIN1_OBJS = main.o $(COMMON_OBJS)
MAIN2_OBJS = mpi_main.o $(COMMON_OBJS)

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(MAIN1_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(TARGET2): $(MAIN2_OBJS)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Common object rules
loader.o: loader.c loader.h
	$(CC) $(CFLAGS) -c loader.c

neural.o: neural.c neural.h
	$(CC) $(CFLAGS) -c neural.c

# Main object rules
main.o: main.c loader.h neural.h
	$(CC) $(CFLAGS) -c main.c

mpi_main.o: mpi_main.c loader.h neural.h
	$(CC) $(CFLAGS) -c mpi_main.c

clean:
	rm -f *.o $(TARGET1) $(TARGET2)

.PHONY: all clean

# end
