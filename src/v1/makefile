CC = gcc
CFLAGS = -Wall -O2 -pg# Profiling enabled with -pg

EXE = nn.exe
SRC = nn.c

all: $(EXE) run profile

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	gprof $(EXE) gmon.out > profile.txt

clean:
	rm -f $(EXE) gmon.out profile.txt profile.png

