#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include <time.h>

#define MAX_CITIES 20

// Structure to represent state of path being built
typedef struct {
    int path[MAX_CITIES + 1];  // Path including return to origin
    int length;                // Current path length
    bool visited[MAX_CITIES];  // Cities visited so far
    int num_cities_visited;    // Number of cities in current path
} PathState;

// Tags for MPI messages
#define WORK_TAG 1
#define RESULT_TAG 2
#define TERMINATE_TAG 3
#define BEST_SOLUTION_TAG 4

// Global variables
int num_cities;
int distances[MAX_CITIES][MAX_CITIES];
int best_distance = INT_MAX;
int best_path[MAX_CITIES + 1];
int rank, size;

// Function prototypes
void read_distances(const char *filename);
void master_process();
void worker_process();
void branch_and_bound(PathState state, int current_best);
int lower_bound(PathState state);
void update_best_solution(PathState state);
void print_solution();

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <distance_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Read the distance matrix
    read_distances(argv[1]);
    
    // Broadcast the distance matrix and number of cities to all processes
    MPI_Bcast(&num_cities, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(distances, MAX_CITIES * MAX_CITIES, MPI_INT, 0, MPI_COMM_WORLD);
    
    double start_time = MPI_Wtime();
    
    // Branch into master and worker processes
    if (rank == 0) {
        master_process();
    } else {
        worker_process();
    }
    
    // Print solution (only master)
    if (rank == 0) {
        double end_time = MPI_Wtime();
        print_solution();
        printf("Execution time: %.2f seconds\n", end_time - start_time);
    }
    
    MPI_Finalize();
    return 0;
}

// Read distance matrix from file
void read_distances(const char *filename) {
    FILE *file;
    
    if (rank == 0) {
        file = fopen(filename, "r");
        if (!file) {
            printf("Error opening file %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Read number of cities
        if (fscanf(file, "%d", &num_cities) != 1) {
            printf("Error reading number of cities\n");
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (num_cities > MAX_CITIES) {
            printf("Too many cities. Maximum supported: %d\n", MAX_CITIES);
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Read distance matrix
        for (int i = 0; i < num_cities; i++) {
            for (int j = 0; j < num_cities; j++) {
                if (fscanf(file, "%d", &distances[i][j]) != 1) {
                    printf("Error reading distance at position (%d, %d)\n", i, j);
                    fclose(file);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
        
        fclose(file);
        
        printf("Read %d cities from file\n", num_cities);
    }
}

// Master process coordinates the work
void master_process() {
    int workers_done = 0;
    MPI_Status status;
    
    // Initial work distribution: assign starting cities to workers
    // We assign each worker a different starting city to explore from
    int starting_city = 0;
    int worker;
    
    for (worker = 1; worker < size && starting_city < num_cities; worker++, starting_city++) {
        MPI_Send(&starting_city, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
    }
    
    // Process remaining starting cities (if more cities than workers)
    while (starting_city < num_cities) {
        // Receive result from any worker
        int result_path[MAX_CITIES + 1];
        int result_distance;
        
        MPI_Recv(&result_distance, 1, MPI_INT, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(result_path, MAX_CITIES + 1, MPI_INT, status.MPI_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
        
        // Update best solution if better
        if (result_distance < best_distance) {
            best_distance = result_distance;
            memcpy(best_path, result_path, sizeof(int) * (MAX_CITIES + 1));
            
            // Broadcast new best solution to all workers
            MPI_Bcast(&best_distance, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(best_path, MAX_CITIES + 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // Assign new work to worker
        MPI_Send(&starting_city, 1, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
        starting_city++;
    }
    
    // Collect results from workers still processing
    while (workers_done < size - 1) {
        int result_path[MAX_CITIES + 1];
        int result_distance;
        
        MPI_Recv(&result_distance, 1, MPI_INT, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(result_path, MAX_CITIES + 1, MPI_INT, status.MPI_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
        
        // Update best solution if better
        if (result_distance < best_distance) {
            best_distance = result_distance;
            memcpy(best_path, result_path, sizeof(int) * (MAX_CITIES + 1));
            
            // Broadcast new best solution to all workers
            MPI_Bcast(&best_distance, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(best_path, MAX_CITIES + 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // Send termination message
        int terminate = -1;
        MPI_Send(&terminate, 1, MPI_INT, status.MPI_SOURCE, TERMINATE_TAG, MPI_COMM_WORLD);
        workers_done++;
    }
}

// Worker process executes branch and bound algorithm
void worker_process() {
    MPI_Status status;
    int starting_city;
    bool working = true;
    
    while (working) {
        // Receive work or termination message
        MPI_Recv(&starting_city, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        if (status.MPI_TAG == TERMINATE_TAG) {
            working = false;
            continue;
        }
        
        // Initialize path state
        PathState initial_state;
        memset(&initial_state, 0, sizeof(PathState));
        
        // Set starting city
        initial_state.path[0] = starting_city;
        initial_state.visited[starting_city] = true;
        initial_state.num_cities_visited = 1;
        
        // Run branch and bound from this starting city
        branch_and_bound(initial_state, best_distance);
        
        // Send result back to master
        MPI_Send(&best_distance, 1, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);
        MPI_Send(best_path, MAX_CITIES + 1, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);
        
        // Check for updated best solution from master
        int flag;
        MPI_Iprobe(0, BEST_SOLUTION_TAG, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            MPI_Recv(&best_distance, 1, MPI_INT, 0, BEST_SOLUTION_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(best_path, MAX_CITIES + 1, MPI_INT, 0, BEST_SOLUTION_TAG, MPI_COMM_WORLD, &status);
        }
    }
}

// Branch and bound algorithm
void branch_and_bound(PathState state, int current_best) {
    // If all cities are visited, complete the tour
    if (state.num_cities_visited == num_cities) {
        // Add distance back to origin
        int origin = state.path[0];
        int last_city = state.path[state.num_cities_visited - 1];
        state.length += distances[last_city][origin];
        state.path[state.num_cities_visited] = origin;
        
        // Update best solution if better
        if (state.length < current_best) {
            update_best_solution(state);
        }
        return;
    }
    
    // Calculate lower bound for current partial path
    int lb = lower_bound(state);
    
    // Prune if lower bound exceeds current best solution
    if (lb >= current_best) {
        return;
    }
    
    // Try adding each unvisited city
    for (int next_city = 0; next_city < num_cities; next_city++) {
        if (!state.visited[next_city]) {
            // Create new state with next_city added
            PathState new_state = state;
            int last_city = new_state.path[new_state.num_cities_visited - 1];
            new_state.length += distances[last_city][next_city];
            new_state.path[new_state.num_cities_visited] = next_city;
            new_state.visited[next_city] = true;
            new_state.num_cities_visited++;
            
            // Recursively explore this path
            branch_and_bound(new_state, current_best);
            
            // Check for updated best solution from master periodically
            if (new_state.num_cities_visited % 5 == 0) {
                MPI_Status status;
                int flag;
                MPI_Iprobe(0, BEST_SOLUTION_TAG, MPI_COMM_WORLD, &flag, &status);
                if (flag) {
                    MPI_Recv(&current_best, 1, MPI_INT, 0, BEST_SOLUTION_TAG, MPI_COMM_WORLD, &status);
                    MPI_Recv(best_path, MAX_CITIES + 1, MPI_INT, 0, BEST_SOLUTION_TAG, MPI_COMM_WORLD, &status);
                }
            }
        }
    }
}

// Calculate lower bound for current partial path
int lower_bound(PathState state) {
    int lb = state.length;
    
    // For each unvisited city, add minimum outgoing edge
    for (int city = 0; city < num_cities; city++) {
        if (!state.visited[city]) {
            int min_edge = INT_MAX;
            for (int j = 0; j < num_cities; j++) {
                if (city != j && distances[city][j] < min_edge) {
                    min_edge = distances[city][j];
                }
            }
            lb += min_edge;
        }
    }
    
    // Add minimum edge back to origin from unvisited cities
    if (state.num_cities_visited < num_cities) {
        int origin = state.path[0];
        int min_return = INT_MAX;
        for (int city = 0; city < num_cities; city++) {
            if (!state.visited[city] && distances[city][origin] < min_return) {
                min_return = distances[city][origin];
            }
        }
        lb += min_return;
    }
    
    return lb;
}

// Update best solution (local to this process)
void update_best_solution(PathState state) {
    best_distance = state.length;
    memcpy(best_path, state.path, sizeof(int) * (num_cities + 1));
}

// Print the best solution found
void print_solution() {
    printf("Best TSP tour found: ");
    for (int i = 0; i <= num_cities; i++) {
        printf("%d ", best_path[i]);
    }
    printf("\nTotal distance: %d\n", best_distance);
}