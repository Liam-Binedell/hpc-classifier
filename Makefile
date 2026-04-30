##
# Federated Learning
#
# @file
# @version 0.1

CC = gcc
CFLAGS = -Wall -O2
LDLIBS = -lm

TARGET = classifier
SOURCES = main.c loader.c neural.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean

# end
