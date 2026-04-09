CC := nvcc
CUDAFLAGS := -arch=sm_120
INCLUDE := -Iinclude

all:\
	bin/main.o\
	bin/kernels.o\
	bin/kernels/naive.o\
	bin/kernels/naive_correct.o\
	bin/kernels/coalesced.o\
	bin/kernels/tiled.o
	$(CC) $(CUDAFLAGS) $? -o bin/a

bin/%.o: src/%.cu
	mkdir -p $(dir $@)
	$(CC) -c $(CUDAFLAGS) $(INCLUDE) $< -o $@

clean:
	rm -rf bin

