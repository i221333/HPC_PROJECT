NVCC      = nvcc
CFLAGS  = -O2 -lcudart -lcurand -lcublas -lm -pg# -pg enables gprof profiling
CUDA_ARCH := -gencode arch=compute_86,code=sm_86

EXE = nn.exe
SRC = nn.cu

all: $(EXE) run profile

$(EXE): $(SRC)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $(EXE) $(SRC)

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	gprof $(EXE) gmon.out > profile.txt

clean:
	rm -f $(EXE) gmon.out profile.txt
