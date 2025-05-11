/*
 * This program simulates a basic life engine on a grid using CUDA for parallel processing.
 * The grid represents a world where organisms, composed of MOUTH and PHOTO cells, interact with FOOD cells.
 * - PHOTO cells can spawn FOOD in adjacent empty cells.
 * - MOUTH cells can eat adjacent FOOD cells, increasing the organism's food count.
 * - Organisms can reproduce into empty cells based on food consumed.
 * The simulation runs for a specified number of iterations, updating the grid state each time.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

// --- Constants ---

// Grid size (N x N)
#define N 100

// Cell types
#define EMPTY 0  // Represents an empty cell
#define MOUTH 1  // Mouth cell of an organism, consumes food
#define PHOTO 2  // Photosynthetic cell of an organism, produces food
#define FOOD 3   // Food cell, can be eaten by MOUTH cells

// Probability that a PHOTO cell spawns food in an adjacent empty cell
#define PHOTO_FOOD_PROB 0.25f

// Probability factor for organism reproduction based on food consumed
#define FOOD_REPRO_PROB 0.1f

// Maximum number of organisms allowed in the simulation
#define MAX_ORGANISMS 10000

// --- CUDA Kernel Functions ---

/**
 * Initializes random number generator states for each cell in the grid.
 * Each thread initializes its own curandState for independent random number generation.
 * 
 * @param states Array of curandState objects, one per cell
 * @param seed Seed value for random number generation
 */
__global__ void initRNG(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x;
    if (idx < N * N) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/**
 * Sets all elements of an array to 1.0f.
 * Used to reset probability arrays before updating them.
 * 
 * @param array The array to set (e.g., probability of not spawning)
 * @param n The size of the grid (N)
 */
__global__ void setToOne(float* array, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        array[row * n + col] = 1.0f;
    }
}

/**
 * Updates the probability of not spawning food in each cell based on adjacent PHOTO cells.
 * For each empty cell, the probability decreases (multiplied by 1 - PHOTO_FOOD_PROB) for each neighboring PHOTO cell.
 * 
 * @param grid The current grid state
 * @param prob_not_spawn Array storing the probability of not spawning food
 * @param n The size of the grid (N)
 */
__global__ void updateFoodProbs(int* grid, float* prob_not_spawn, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float prob = 1.0f;
        if (col > 0 && grid[row * n + (col - 1)] == PHOTO) prob *= (1.0f - PHOTO_FOOD_PROB);
        if (col < n - 1 && grid[row * n + (col + 1)] == PHOTO) prob *= (1.0f - PHOTO_FOOD_PROB);
        if (row > 0 && grid[(row - 1) * n + col] == PHOTO) prob *= (1.0f - PHOTO_FOOD_PROB);
        if (row < n - 1 && grid[(row + 1) * n + col] == PHOTO) prob *= (1.0f - PHOTO_FOOD_PROB);
        prob_not_spawn[row * n + col] = prob;
    }
}

/**
 * Updates the probability of not spawning new organisms based on food consumed by MOUTH cells.
 * For each MOUTH cell, a multiplier is calculated from the organism's food count and applied to empty cells three steps away.
 * 
 * @param grid The current grid state
 * @param id_grid Array of organism IDs for each cell
 * @param organism_food Array storing food count for each organism
 * @param prob_not_spawn Array storing the probability of not spawning organisms
 * @param n The size of the grid (N)
 */
__global__ void updateSpawnProbs(int* grid, int* id_grid, int* organism_food, float* prob_not_spawn, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * n + col;

    if (row < n && col < n && grid[idx] == MOUTH) {
        int organism_id = id_grid[idx];
        float multiplier = powf(1.0f - FOOD_REPRO_PROB, organism_food[organism_id]);

        if (col > 2 && grid[row * n + (col - 3)] == EMPTY) prob_not_spawn[row * n + (col - 3)] *= multiplier;
        if (col < n - 3 && grid[row * n + (col + 3)] == EMPTY) prob_not_spawn[row * n + (col + 3)] *= multiplier;
        if (row > 2 && grid[(row - 3) * n + col] == EMPTY) prob_not_spawn[(row - 3) * n + col] *= multiplier;
        if (row < n - 3 && grid[(row + 3) * n + col] == EMPTY) prob_not_spawn[(row + 3) * n + col] *= multiplier;
    }
}

/**
 * Spawns FOOD in empty cells based on calculated probabilities.
 * A random number is generated for each empty cell; if it's less than (1 - prob_not_spawn), FOOD is spawned.
 * 
 * @param grid The current grid state
 * @param prob_not_spawn Array of probabilities of not spawning food
 * @param states Random number generator states
 * @param n The size of the grid (N)
 */
__global__ void spawnFood(int* grid, float* prob_not_spawn, curandState* states, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int idx = row * n + col;
        if (grid[idx] == EMPTY) {
            float rand = curand_uniform(&states[idx]);
            if (rand < (1.0f - prob_not_spawn[idx])) {
                grid[idx] = FOOD;
            }
        }
    }
}

/**
 * Attempts organism reproduction into empty cells based on probabilities.
 * If cells at (row-1, col-1) and (row+1, col+1) are empty, a new organism may form with PHOTO cells there and a MOUTH at (row, col).
 * 
 * @param grid The current grid state
 * @param id_grid Array of organism IDs for each cell
 * @param next_id Pointer to the next available organism ID
 * @param prob_not_spawn Array of probabilities of not spawning organisms
 * @param states Random number generator states
 * @param n The size of the grid (N)
 */
__global__ void reproduce(int* grid, int* id_grid, int* next_id, float* prob_not_spawn, curandState* states, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > 0 && row < n - 1 && col > 0 && col < n - 1) {
        if (grid[(row - 1) * n + (col - 1)] == EMPTY && grid[(row + 1) * n + (col + 1)] == EMPTY) {
            float rand = curand_uniform(&states[row * n + col]);
            if (rand < (1.0f - prob_not_spawn[row * n + col])) {
                int new_id = atomicAdd(next_id, 1);
                grid[(row - 1) * n + (col - 1)] = PHOTO;
                grid[row * n + col] = MOUTH;
                grid[(row + 1) * n + (col + 1)] = PHOTO;
                id_grid[row * n + col] = new_id;
            }
        }
    }
}

/**
 * Allows MOUTH cells to eat adjacent FOOD cells.
 * For each FOOD cell, if an adjacent cell is a MOUTH, the FOOD is consumed, and the organism's food count increases.
 * 
 * @param grid The current grid state
 * @param id_grid Array of organism IDs for each cell
 * @param organism_food Array storing food count for each organism
 * @param n The size of the grid (N)
 */
__global__ void eatFood(int* grid, int* id_grid, int* organism_food, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * n + col;

    if (row < n && col < n && grid[idx] == FOOD) {
        if (col > 0 && grid[row * n + (col - 1)] == MOUTH) {
            grid[idx] = EMPTY;
            int id = id_grid[row * n + (col - 1)];
            atomicAdd(&organism_food[id], 1);
        } else if (col < n - 1 && grid[row * n + (col + 1)] == MOUTH) {
            grid[idx] = EMPTY;
            int id = id_grid[row * n + (col + 1)];
            atomicAdd(&organism_food[id], 1);
        } else if (row > 0 && grid[(row - 1) * n + col] == MOUTH) {
            grid[idx] = EMPTY;
            int id = id_grid[(row - 1) * n + col];
            atomicAdd(&organism_food[id], 1);
        } else if (row < n - 1 && grid[(row + 1) * n + col] == MOUTH) {
            grid[idx] = EMPTY;
            int id = id_grid[(row + 1) * n + col];
            atomicAdd(&organism_food[id], 1);
        }
    }
}

// --- Host Function ---

/**
 * Prints the current state of the grid to the console.
 * Uses symbols: ' ' (EMPTY), 'M' (MOUTH), 'P' (PHOTO), 'F' (FOOD).
 * 
 * @param grid The grid to print
 * @param width The width of the grid (N)
 * @param height The height of the grid (N)
 */
void printGrid(int* grid, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int val = grid[y * width + x];
            char symbol = (val == EMPTY) ? ' ' : (val == MOUTH) ? 'M' : (val == PHOTO) ? 'P' : 'F';
            printf("%c ", symbol);
        }
        printf("\n");
    }
}

// --- Main Simulation ---

int main() {
    // Allocate host memory for the grid and organism IDs
    int* grid = (int*)calloc(N * N, sizeof(int));
    int* organism_ids = (int*)calloc(N * N, sizeof(int));

    // Initialize the grid with a single organism at the center
    grid[N * 49 + 49] = PHOTO;    // Top-left PHOTO cell
    grid[N * 50 + 50] = MOUTH;    // Central MOUTH cell
    grid[N * 51 + 51] = PHOTO;    // Bottom-right PHOTO cell
    organism_ids[50 * N + 50] = 1; // Assign organism ID 1 to the MOUTH cell

    // Declare device pointers
    int *grid_d;            // Device grid
    float *food_probs_d;    // Device array for food spawn probabilities
    float *spawn_probs_d;   // Device array for organism spawn probabilities
    curandState *states_d;  // Device array for RNG states
    int *organism_id_d;     // Device array for organism IDs
    int *organism_food_d;   // Device array for organism food counts
    int *next_id_d;         // Device pointer for the next organism ID

    // Allocate memory on the GPU
    cudaMalloc(&grid_d, N * N * sizeof(int));
    cudaMalloc(&food_probs_d, N * N * sizeof(float));
    cudaMalloc(&spawn_probs_d, N * N * sizeof(float));
    cudaMalloc(&states_d, N * N * sizeof(curandState));
    cudaMalloc(&organism_id_d, N * N * sizeof(int));
    cudaMalloc(&organism_food_d, MAX_ORGANISMS * sizeof(int));
    cudaMalloc(&next_id_d, sizeof(int));

    // Copy initial data from host to device
    cudaMemcpy(grid_d, grid, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(organism_id_d, organism_ids, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(organism_food_d, 0, MAX_ORGANISMS * sizeof(int)); // Initialize food counts to zero

    // Set the next available organism ID
    int next_id = 2;
    cudaMemcpy(next_id_d, &next_id, sizeof(int), cudaMemcpyHostToDevice);

    // Define CUDA block and grid dimensions
    dim3 blockDim(10, 10);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Initialize RNG states on the device
    initRNG<<<gridDim, blockDim>>>(states_d, time(NULL));
    cudaDeviceSynchronize();

    // Run the simulation for a fixed number of iterations
    int num_iterations = 100;
    for (int t = 0; t < num_iterations; t++) {
        // Reset organism spawn probabilities
        setToOne<<<gridDim, blockDim>>>(spawn_probs_d, N);

        // Update food spawn probabilities based on PHOTO cells
        updateFoodProbs<<<gridDim, blockDim>>>(grid_d, food_probs_d, N);

        // Spawn food in empty cells
        spawnFood<<<gridDim, blockDim>>>(grid_d, food_probs_d, states_d, N);

        // Allow MOUTH cells to consume adjacent FOOD
        eatFood<<<gridDim, blockDim>>>(grid_d, organism_id_d, organism_food_d, N);

        // Update organism spawn probabilities based on food consumed
        updateSpawnProbs<<<gridDim, blockDim>>>(grid_d, organism_id_d, organism_food_d, spawn_probs_d, N);

        // Attempt organism reproduction
        reproduce<<<gridDim, blockDim>>>(grid_d, organism_id_d, next_id_d, spawn_probs_d, states_d, N);

        // Ensure all kernels complete before proceeding
        cudaDeviceSynchronize();

        // Copy the updated grid back to the host
        cudaMemcpy(grid, grid_d, N * N * sizeof(int), cudaMemcpyDeviceToHost);

        // Display the current state
        printf("Iteration %d:\n", t);
        printGrid(grid, N, N);

        // Pause for user input to view each iteration
        getchar();
    }

    // Free GPU memory
    cudaFree(grid_d);
    cudaFree(food_probs_d);
    cudaFree(spawn_probs_d);
    cudaFree(states_d);
    cudaFree(organism_id_d);
    cudaFree(organism_food_d);
    cudaFree(next_id_d);

    // Free host memory
    free(grid);
    free(organism_ids);

    return 0;
}
