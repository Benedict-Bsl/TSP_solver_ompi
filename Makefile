# CC = mpicc
# CFLAGS = -Wall -O3
# TARGET = solving_traveller

# all: $(TARGET)

# $(TARGET): solving_traveller.c
# 	$(CC) $(CFLAGS) -o $(TARGET) solving_traveller.c

# clean:
# 	rm -f $(TARGET)

# run:
# 	mpirun -np 4 ./$(TARGET) known_distances_sample.txt

# .PHONY: all clean run


CC = mpicc
CFLAGS = -Wall -O3 -Wno-unused-result
TARGET = solving_traveller

all: $(TARGET)

$(TARGET): solving_traveller.c
	$(CC) $(CFLAGS) -o $(TARGET) solving_traveller.c

debug: CFLAGS = -Wall -g -Wno-unused-result
debug: $(TARGET)

clean:
	rm -f $(TARGET)

run:
	mpirun -np 4 ./$(TARGET) known_distances_sample.txt

test:
	# Create a small test case for quick verification
	echo "4" > test_distances.txt
	echo "0 10 15 20" >> test_distances.txt
	echo "10 0 35 25" >> test_distances.txt
	echo "15 35 0 30" >> test_distances.txt
	echo "20 25 30 0" >> test_distances.txt
	mpirun -np 2 ./$(TARGET) test_distances.txt
	rm test_distances.txt

.PHONY: all clean run test debug