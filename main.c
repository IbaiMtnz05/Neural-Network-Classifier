#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <errno.h>
#include <unistd.h> // For getcwd
#include <SDL2/SDL.h> // Para la visualización
#include <time.h>

// Definición de constantes para el visualizador
#define WINDOW_WIDTH 560  // 28*20
#define WINDOW_HEIGHT 560 // 28*20
#define PIXEL_SIZE 20     // Cada pixel MNIST se mostrará como un cuadrado de 20x20

// Estructura para el visualizador
typedef struct {
    SDL_Window *window;
    SDL_Renderer *renderer;
    int current_image;
    int running;
} Viewer;

// Add this structure after the existing typedefs
typedef struct {
    clock_t start;
    clock_t end;
    double cpu_time;
    const char* operation;
} TimingInfo;

// Function prototypes
int control_errores(const char *checkFile);
int read_matrix(double **mat, char *file, int nrows, int ncols, int fac);
int read_vector(double *vect, char *file, int nrows);
void print_matrix(double **mat, int nrows, int ncols, int offset_row, int offset_col);
void load_data(char *path);
void unload_data(void);
double** mat_mul(double **input, int input_rows, int input_cols, double **weights, int weight_cols);
double** sum_vect(double **matrix, double *vector, int nrows, int ncols);
double** relu(double **matrix, int nrows, int ncols);
int* argmax(double **matrix, int rows, int cols);
void free_matrix(double **matrix, int rows);
int* forward_pass(double **data);
char *siguiente_token(char *buffer);
void view_mnist_images(double **data, int num_images);
double error_log(int *predictions, double *actual_digits, int num_samples, int max_errors_to_log);

// Global variables
static double **data;
int data_nrows;
int data_ncols = 784;
char *my_path; // TO DO auto detection
int seed = 3;
// Weight matrices dimensions (as given):
// mat1: 784 x 200, mat2: 200 x 100, mat3: 100 x 50, mat4: 50 x 10.
int matrices_rows[4]    = {784, 200, 100, 50};
int matrices_columns[4] = {200, 100, 50, 10};
// Bias vectors dimensions: match output columns of each layer.
int vector_rows[4] = {200, 100, 50, 10};

char *str;  // for building file paths

static double *digits;
static double **mat1;
static double **mat2;
static double **mat3;
static double **mat4;
static double *vec1;
static double *vec2;
static double *vec3;
static double *vec4;

// Función para visualizar las imágenes MNIST
void view_mnist_images(double **data, int num_images) {
    if (data == NULL || num_images <= 0) {
        fprintf(stderr, "Error: Datos no válidos para visualizar\n");
        return;
    }
    
    // Inicializar SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "Error al inicializar SDL: %s\n", SDL_GetError());
        return;
    }
    
    // Crear estructura del visualizador
    Viewer viewer;
    viewer.current_image = 0;
    viewer.running = 1;
    
    // Crear ventana
    viewer.window = SDL_CreateWindow(
        "Visualizador MNIST",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN
    );
    
    if (!viewer.window) {
        fprintf(stderr, "Error al crear ventana: %s\n", SDL_GetError());
        SDL_Quit();
        return;
    }
    
    // Crear renderer
    viewer.renderer = SDL_CreateRenderer(
        viewer.window,
        -1,
        SDL_RENDERER_ACCELERATED
    );
    
    if (!viewer.renderer) {
        fprintf(stderr, "Error al crear renderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(viewer.window);
        SDL_Quit();
        return;
    }
    
    // Bucle principal
    SDL_Event event;
    while (viewer.running) {
        // Procesar eventos
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    viewer.running = 0;
                    break;
                case SDL_KEYDOWN:
                    switch (event.key.keysym.sym) {
                        case SDLK_RIGHT: // Flecha derecha - siguiente imagen
                            viewer.current_image = (viewer.current_image + 1) % num_images;
                            break;
                        case SDLK_LEFT: // Flecha izquierda - imagen anterior
                            viewer.current_image = (viewer.current_image - 1 + num_images) % num_images;
                            break;
                        case SDLK_ESCAPE: // Escape - salir
                            viewer.running = 0;
                            break;
                        default:
                            break;
                    }
                    break;
            }
        }
        
        // Limpiar pantalla
        SDL_SetRenderDrawColor(viewer.renderer, 0, 0, 0, 255);
        SDL_RenderClear(viewer.renderer);
        
        // Renderizar imagen actual
        double *current_data = data[viewer.current_image];
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int index = i * 28 + j;
                // El valor está en el rango [0, 255], donde 0 es negro y 255 es blanco
                int pixel_value = (int)(current_data[index]);
                
                // Configurar color (escala de grises)
                SDL_SetRenderDrawColor(viewer.renderer, pixel_value, pixel_value, pixel_value, 255);
                
                // Dibujar píxel ampliado
                SDL_Rect rect = {
                    j * PIXEL_SIZE,
                    i * PIXEL_SIZE,
                    PIXEL_SIZE,
                    PIXEL_SIZE
                };
                SDL_RenderFillRect(viewer.renderer, &rect);
            }
        }
        
        // Mostrar información de imagen actual
        char info_text[100];
        sprintf(info_text, "Imagen %d/%d - Use flechas izq/der para navegar, ESC para salir", 
                viewer.current_image + 1, num_images);
        
        // Renderizar en pantalla
        SDL_RenderPresent(viewer.renderer);
        
        // Pequeña pausa para no consumir demasiada CPU
        SDL_Delay(10);
    }
    
    // Liberar recursos
    SDL_DestroyRenderer(viewer.renderer);
    SDL_DestroyWindow(viewer.window);
    SDL_Quit();
}

// Debug
// Print a matrix for debugging purposes.
void debug_print_matrix(double **mat, int nrows, int ncols, const char *name) {
    printf("\n%s (%d x %d):\n", name, nrows, ncols);
    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            printf("%8.3f ", mat[row][col]);
        }
        printf("\n");
    }
}

// Print a vector for debugging purposes.
void debug_print_vector(double *vec, int nrows, const char *name) {
    printf("\n%s (%d):\n", name, nrows);
    for (int i = 0; i < nrows; i++) {
        printf("%8.3f ", vec[i]);
    }
    printf("\n");
}

// Helper for tokenization - FIXED to better handle CSV parsing
char *siguiente_token(char *buffer) {
    static char *last_ptr = NULL;
    if (buffer != NULL) {
        last_ptr = buffer;
        return strtok(last_ptr, " ,\n\r");
    } else {
        return strtok(NULL, " ,\n\r");
    }
}

// Read a CSV file into a 2D matrix - FIXED to handle different CSV formats
int read_matrix(double **mat, char *file, int nrows, int ncols, int fac) {
    printf("\nRead matrix from file: %s\n", file);
    char buffer[1024 * 10]; // Increased buffer size for larger lines
    //char *buffer =(double*) malloc(ncols*nrows * sizeof(double)); // Malloc pero no funciona
    FILE *fstream = fopen(file, "r");
    if (fstream == NULL || control_errores(file) != 0) {
        printf("Error opening file: %s\n", file);
        return 1;
    }
    
    for (int row = 0; row < nrows; row++) {
        if (fgets(buffer, sizeof(buffer), fstream) == NULL) {
            printf("Warning: Reached end of file at row %d/%d\n", row, nrows);
            break;
        }
        
        // Remove any trailing newline or carriage return
        size_t len = strlen(buffer);
        if (len > 0 && (buffer[len-1] == '\n' || buffer[len-1] == '\r')) {
            buffer[len-1] = '\0';
            if (len > 1 && buffer[len-2] == '\r') {
                buffer[len-2] = '\0';
            }
        }
        
        char *token = siguiente_token(buffer);
        for (int col = 0; col < ncols; col++) {
            if (token) {
                mat[row][col] = strtod(token, NULL) * fac;
                token = siguiente_token(NULL);
            } else {
                printf("Warning: Missing value at row %d, col %d\n", row, col);
                mat[row][col] = 0.0; // Use 0 instead of -1 for missing values
            }
        }
    }
    fclose(fstream);
    return 0;
}

// Read a CSV file into a vector - FIXED to handle different CSV formats
int read_vector(double *vect, char *file, int nrows) {
    printf("\nRead vector from file: %s\n", file);
    FILE *fstream = fopen(file, "r");
    if (fstream == NULL || control_errores(file) != 0) {
        printf("Error opening file: %s\n", file);
        return 1;
    }
    
    char buffer[1024];
    for (int i = 0; i < nrows; i++) {
        if (fgets(buffer, sizeof(buffer), fstream) == NULL) {
            printf("Warning: Reached end of file at row %d/%d\n", i, nrows);
            break;
        }
        
        // Remove newline and carriage return
        size_t len = strlen(buffer);
        if (len > 0 && (buffer[len-1] == '\n' || buffer[len-1] == '\r')) {
            buffer[len-1] = '\0';
            if (len > 1 && buffer[len-2] == '\r') {
                buffer[len-2] = '\0';
            }
        }
        
        // Handle possible comma-separated values (take first value)
        char *comma = strchr(buffer, ',');
        if (comma) *comma = '\0';
        
        vect[i] = strtod(buffer, NULL);
    }
    fclose(fstream);
    return 0;
}

void print_matrix(double **mat, int nrows, int ncols, int offset_row, int offset_col) {
    if (!mat) {
        printf("Error: La matriz no está inicializada.\n");
        return;
    }
    printf("\nMatriz (%d x %d) desde offset (%d, %d):\n", nrows, ncols, offset_row, offset_col);
    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            printf("%8.3f ", mat[row + offset_row][col + offset_col]);
        }
        printf("\n");
    }
}

// Load all data and model parameters - FIXED to better handle file loading
void load_data(char *path) {
    // Allocate buffer for file paths.
    str = malloc(256); // Increased buffer size for longer paths
    
    // Load digits (the ground-truth labels)
    printf("Cargando digits...\n");
    digits = malloc(data_nrows * sizeof(double));
    if (!digits) {
        fprintf(stderr, "Error: No se pudo asignar memoria para digits\n");
        exit(1);
    }
    
    // Initialize digits array with -1 to detect loading issues
    for (int i = 0; i < data_nrows; i++) {
        digits[i] = -1;
    }
    
    sprintf(str, "%scsvs/digits.csv", path);
    if (read_vector(digits, str, data_nrows) != 0) {
        fprintf(stderr, "Error: No se pudieron cargar los digits\n");
        exit(1);
    }
    printf("Digits cargados.\n");

    // Allocate and load input data (as a 2D array).
    printf("Cargando data...\n");
    data = malloc(data_nrows * sizeof(double *));
    if (!data) {
        fprintf(stderr, "Error: No se pudo asignar memoria para data\n");
        exit(1);
    }
    
    for (int i = 0; i < data_nrows; i++) {
        data[i] = malloc(data_ncols * sizeof(double));
        if (!data[i]) {
            fprintf(stderr, "Error: No se pudo asignar memoria para data[%d]\n", i);
            exit(1);
        }
        // Initialize with zeros to detect loading issues
        for (int j = 0; j < data_ncols; j++) {
            data[i][j] = 0;
        }
    }
    
    sprintf(str, "%scsvs/data.csv", path);
    if (read_matrix(data, str, data_nrows, data_ncols, 1) != 0) {
        fprintf(stderr, "Error: No se pudo cargar data.csv\n");
        exit(1);
    }
    
    printf("Data cargada.\n");
    print_matrix(data, 5, 5, 0, 0);
    
    // Check if data is loaded correctly by examining a few values
    int has_nonzero = 0;
    for (int i = 0; i < 5 && i < data_nrows; i++) {
        for (int j = 0; j < 5 && j < data_ncols; j++) {
            if (data[i][j] != 0) {
                has_nonzero = 1;
                break;
            }
        }
        if (has_nonzero) break;
    }
    if (!has_nonzero) {
        printf("Warning: La data parece contener solo ceros. Verificar el formato del archivo CSV\n");
    }
    
    // Load weight matrices.
    // mat1: 784 x 200
    printf("Cargando mat1...\n");
    mat1 = malloc(matrices_rows[0] * sizeof(double *));
    for (int i = 0; i < matrices_rows[0]; i++) {
        mat1[i] = malloc(matrices_columns[0] * sizeof(double));
        for (int j = 0; j < matrices_columns[0]; j++) {
            mat1[i][j] = 0;
        }
    }
    sprintf(str, "%sparameters/weights%d_%d.csv", path, 0, seed);
    read_matrix(mat1, str, matrices_rows[0], matrices_columns[0], 1);
    printf("mat1 cargada.\n");

    // mat2: 200 x 100
    printf("Cargando mat2...\n");
    mat2 = malloc(matrices_rows[1] * sizeof(double *));
    for (int i = 0; i < matrices_rows[1]; i++) {
        mat2[i] = malloc(matrices_columns[1] * sizeof(double));
        for (int j = 0; j < matrices_columns[1]; j++) {
            mat2[i][j] = 0;
        }
    }
    sprintf(str, "%sparameters/weights%d_%d.csv", path, 1, seed);
    read_matrix(mat2, str, matrices_rows[1], matrices_columns[1], 1);
    printf("mat2 cargada.\n");

    // mat3: 100 x 50
    printf("Cargando mat3...\n");
    mat3 = malloc(matrices_rows[2] * sizeof(double *));
    for (int i = 0; i < matrices_rows[2]; i++) {
        mat3[i] = malloc(matrices_columns[2] * sizeof(double));
        for (int j = 0; j < matrices_columns[2]; j++) {
            mat3[i][j] = 0;
        }
    }
    sprintf(str, "%sparameters/weights%d_%d.csv", path, 2, seed);
    read_matrix(mat3, str, matrices_rows[2], matrices_columns[2], 1);
    printf("mat3 cargada.\n");
    
    // mat4: 50 x 10
    printf("Cargando mat4...\n");
    mat4 = malloc(matrices_rows[3] * sizeof(double *));
    for (int i = 0; i < matrices_rows[3]; i++) {
        mat4[i] = malloc(matrices_columns[3] * sizeof(double));
        for (int j = 0; j < matrices_columns[3]; j++) {
            mat4[i][j] = 0;
        }
    }
    sprintf(str, "%sparameters/weights%d_%d.csv", path, 3, seed);
    read_matrix(mat4, str, matrices_rows[3], matrices_columns[3], 1);
    printf("mat4 cargada.\n");

    // Load bias vectors.
    // vec1: dimension 200
    vec1 = malloc(vector_rows[0] * sizeof(double));
    for (int i = 0; i < vector_rows[0]; i++) {
        vec1[i] = 0;
    }
    sprintf(str, "%sparameters/biases%d_%d.csv", path, 0, seed);
    read_vector(vec1, str, vector_rows[0]);
    printf("vec1 cargada.\n");

    // vec2: dimension 100
    vec2 = malloc(vector_rows[1] * sizeof(double));
    for (int i = 0; i < vector_rows[1]; i++) {
        vec2[i] = 0;
    }
    sprintf(str, "%sparameters/biases%d_%d.csv", path, 1, seed);
    read_vector(vec2, str, vector_rows[1]);
    printf("vec2 cargada.\n");

    // vec3: dimension 50
    vec3 = malloc(vector_rows[2] * sizeof(double));
    for (int i = 0; i < vector_rows[2]; i++) {
        vec3[i] = 0;
    }
    sprintf(str, "%sparameters/biases%d_%d.csv", path, 2, seed);
    read_vector(vec3, str, vector_rows[2]);
    printf("vec3 cargada.\n");
    debug_print_vector(vec3, vector_rows[2], "vec3");

    // vec4: dimension 10
    vec4 = malloc(vector_rows[3] * sizeof(double));
    for (int i = 0; i < vector_rows[3]; i++) {
        vec4[i] = 0;
    }
    sprintf(str, "%sparameters/biases%d_%d.csv", path, 3, seed);
    read_vector(vec4, str, vector_rows[3]);
    printf("vec4 cargada.\n");
    debug_print_vector(vec4, vector_rows[3], "vec4");
}

// Free all allocated memory.
void unload_data() {
    free(digits);
    for (int i = 0; i < data_nrows; i++) {
        free(data[i]);
    }
    free(data);
    free_matrix(mat1, matrices_rows[0]);
    free_matrix(mat2, matrices_rows[1]);
    free_matrix(mat3, matrices_rows[2]);
    free_matrix(mat4, matrices_rows[3]);
    free(vec1);
    free(vec2);
    free(vec3);
    free(vec4);
    free(str);
}

void print(void *arg) { printf("Hola, soy %d\n", *(int *)arg); }

// Matrix multiplication:
// - input: (input_rows x input_cols)
// - weights: (input_cols x weight_cols)
// Result: (input_rows x weight_cols)
double** mat_mul(double **input, int input_rows, int input_cols, double **weights, int weight_cols) {
    double **result = malloc(input_rows * sizeof(double *));
    for (int i = 0; i < input_rows; i++) {
        result[i] = malloc(weight_cols * sizeof(double));
    }
    for (int i = 0; i < input_rows; i++) {
        for (int j = 0; j < weight_cols; j++) {
            result[i][j] = 0;
            for (int k = 0; k < input_cols; k++) {
                result[i][j] += input[i][k] * weights[k][j];
            }
        }
    }
    return result;
}

// Add bias vector to every row of the matrix.
double** sum_vect(double **matrix, double *vector, int nrows, int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            matrix[i][j] += vector[j];
        }
    }
    return matrix;
}

// ReLU activation: replace negative values with 0.
double** relu(double **matrix, int nrows, int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (matrix[i][j] < 0)
                matrix[i][j] = 0;
        }
    }
    return matrix;
}

// For each row, return the index of the maximum element.
int* argmax(double **matrix, int rows, int cols) {
    int *predictions = malloc(rows * sizeof(int));
    for (int i = 0; i < rows; i++) {
        double max_val = matrix[i][0];
        int max_idx = 0;
        for (int j = 1; j < cols; j++) {
            if (matrix[i][j] > max_val) {
                max_val = matrix[i][j];
                max_idx = j;
            }
        }
        predictions[i] = max_idx;
    }
    return predictions;
}

// Free a 2D matrix.
void free_matrix(double **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Perform the forward pass through the network.
int* forward_pass(double **data) {
    double **capa0, **capa1, **capa2, **capa3;
    int *predicciones;
    
    printf("\n=== Starting Forward Pass ===\n");
    
    // Layer 0: data (data_nrows x 784) * mat1 (784 x 200)
    printf("\n--- Layer 0 ---\n");
    capa0 = mat_mul(data, data_nrows, data_ncols, mat1, matrices_columns[0]);
    capa0 = sum_vect(capa0, vec1, data_nrows, matrices_columns[0]);
    capa0 = relu(capa0, data_nrows, matrices_columns[0]);
    printf("Layer 0 complete. Output shape: [%d x %d]\n", data_nrows, matrices_columns[0]);
    
    // Layer 1: capa0 (data_nrows x 200) * mat2 (200 x 100)
    printf("\n--- Layer 1 ---\n");
    capa1 = mat_mul(capa0, data_nrows, matrices_columns[0], mat2, matrices_columns[1]);
    capa1 = sum_vect(capa1, vec2, data_nrows, matrices_columns[1]);
    capa1 = relu(capa1, data_nrows, matrices_columns[1]);
    printf("Layer 1 complete. Output shape: [%d x %d]\n", data_nrows, matrices_columns[1]);
    free_matrix(capa0, data_nrows);
    
    // Layer 2: capa1 (data_nrows x 100) * mat3 (100 x 50)
    printf("\n--- Layer 2 ---\n");
    capa2 = mat_mul(capa1, data_nrows, matrices_columns[1], mat3, matrices_columns[2]);
    capa2 = sum_vect(capa2, vec3, data_nrows, matrices_columns[2]);
    capa2 = relu(capa2, data_nrows, matrices_columns[2]);
    printf("Layer 2 complete. Output shape: [%d x %d]\n", data_nrows, matrices_columns[2]);
    free_matrix(capa1, data_nrows);
    
    // Layer 3: capa2 (data_nrows x 50) * mat4 (50 x 10)
    printf("\n--- Layer 3 (Final Layer) ---\n");
    capa3 = mat_mul(capa2, data_nrows, matrices_columns[2], mat4, matrices_columns[3]);
    capa3 = sum_vect(capa3, vec4, data_nrows, matrices_columns[3]);
    capa3 = relu(capa3, data_nrows, matrices_columns[3]);
    printf("Layer 3 complete. Output shape: [%d x %d]\n", data_nrows, matrices_columns[3]);
    free_matrix(capa2, data_nrows);
    
    // Compute predictions using argmax.
    printf("\n--- Computing Final Predictions ---\n");
    predicciones = argmax(capa3, data_nrows, matrices_columns[3]);
    printf("Predictions computed for %d samples\n", data_nrows);
    
    // Print first few predictions.
    printf("\nFirst 100 predictions:\n");
    for (int i = 0; i < 100 && i < data_nrows; i++) {
        printf("Sample %d: Predicted digit %d\n", i, predicciones[i]);
    }
    
    free_matrix(capa3, data_nrows);
    printf("\n=== Forward Pass Complete ===\n");
    
    return predicciones;
}

int control_errores(const char *checkFile) {
    FILE *f = fopen(checkFile, "r");
    if (f == NULL) {
        printf("errno: %d\n", errno);
        printf("Error: %s\n", strerror(errno));
        perror("Houston, tenemos un problema");
        return 1;
    }
    printf("No tenemos problemas con el archivo: %s\n", checkFile);
    fclose(f);
    return 0;
}

double final_result(int *predictions, double *actual_digits, int num_samples) {
    int correct_predictions = 0;
    
    for (int i = 0; i < num_samples; i++) {
        if (predictions[i] == (int)actual_digits[i]) {
            correct_predictions++;
        }
    }
    
    return (correct_predictions / (double)num_samples) * 100.0;
}

double error_log(int *predictions, double *actual_digits, int num_samples, int max_errors_to_log) {
    int total_errors = 0;
    int logged_errors = 0;
    
    printf("\n=== Error Log: Model Prediction Failures ===\n");
    printf("Format: [Line Number] Predicted: X, Actual: Y\n");
    printf("----------------------------------------\n");
    
    // Count errors and log details for misclassified samples
    for (int i = 0; i < num_samples; i++) {
        if (predictions[i] != (int)actual_digits[i]) {
            total_errors++;
            
            // Log error details (up to max_errors_to_log)
            if (logged_errors < max_errors_to_log) {
                printf("[Line %5d] Predicted: %d, Actual: %.0f\n", 
                       i + 1,  // Line number (1-indexed for user readability)
                       predictions[i],  // Model's prediction
                       actual_digits[i]);  // Actual digit
                logged_errors++;
            }
        }
    }
    
    // If there are more errors than we logged, indicate that
    if (total_errors > logged_errors) {
        printf("... and %d more errors not shown\n", total_errors - logged_errors);
    }
    
    // Calculate and return error rate
    double error_rate = (total_errors / (double)num_samples) * 100.0;
    printf("\nSummary: %d errors out of %d samples (%.2f%% error rate)\n", 
           total_errors, num_samples, error_rate);
    
    return error_rate;
}

// Add these functions before main()
void start_timing(TimingInfo* timing, const char* operation) {
    timing->operation = operation;
    timing->start = clock();
}

void end_timing(TimingInfo* timing) {
    timing->end = clock();
    timing->cpu_time = ((double) (timing->end - timing->start)) / CLOCKS_PER_SEC;
}

void print_timing(TimingInfo* timing) {
    printf("│ %-36s│ %11.4f s │\n", timing->operation, timing->cpu_time);
}

void print_timing_header() {
    printf("┌─────────────────────────────────────┬───────────────┐\n");
    printf("│ Operation                           │   Time (s)    │\n");
    printf("├─────────────────────────────────────┼───────────────┤\n");
}

void print_timing_footer() {
    printf("└─────────────────────────────────────┴───────────────┘\n");
}

// Modify main() function:
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("El programa debe tener al menos dos argumentos, la cantidad de procesos que se van a generar o el modo de ejecucion\n");
        exit(1);
    }

    TimingInfo timings[10];  // Array to store timing information
    int timing_index = 0;
    double total_time = 0.0;
    
    // Use a smaller dataset size for testing if full dataset has issues
    // For full MNIST, this would be 60000. Using smaller size for testing.
    data_nrows = 10000; // Start with a smaller subset for testing
    char *my_path = getcwd(NULL, 0);
    // Verify path and adjust if needed
    struct {
        const char* path;
        int tested;
    } paths[] = {
        {my_path, 0},
        {"./", 0},
        {"../", 0},
        {NULL, 0}
    };
    
    char test_path[256];
    int path_found = 0;
    
    for (int i = 0; paths[i].path != NULL; i++) {
        sprintf(test_path, "%scsvs/data.csv", paths[i].path);
        FILE *test = fopen(test_path, "r");
        if (test) {
            fclose(test);
            my_path = strdup(paths[i].path);
            path_found = 1;
            printf("Using path: %s\n", my_path);
            break;
        }
    }
    
    if (!path_found) {
        printf("No se ha encontrado el archivo data.csv en ninguna de las rutas probadas.\n");
        printf("Por favor, especifique la ruta correcta en la variable 'my_path'.\n");
        return 1;
    }
    
    // Start timing data loading
    start_timing(&timings[timing_index], "Data Loading");
    load_data(my_path);
    end_timing(&timings[timing_index++]);
    
    // Verifiquemos si los datos se cargaron correctamente
    int datos_validos = 1;
    for (int i = 0; i < 10 && i < data_nrows; i++) {
        int zeros_count = 0;
        for (int j = 0; j < data_ncols; j++) {
            if (data[i][j] == 0) zeros_count++;
        }
        if (zeros_count == data_ncols) {
            printf("Warning: La fila %d contiene solo ceros.\n", i);
            datos_validos = 0;
        }
    }
    
    if (!datos_validos) {
        printf("Warning: Posible problema con la lectura de data.csv. Revise el formato del archivo.\n");
    }
    
    // Time the MNIST viewer
    start_timing(&timings[timing_index], "MNIST Viewing time");
    // Mostrar el visualizador de imágenes MNIST
    printf("\n=== Iniciando Visualizador de Imágenes MNIST ===\n");
    printf("Use las flechas izquierda/derecha para navegar entre imágenes\n");
    printf("Presione ESC para cerrar el visualizador y continuar con el programa\n");
    view_mnist_images(data, data_nrows);
    end_timing(&timings[timing_index++]);
    printf("\n=== Visualizador cerrado, continuando con el programa ===\n");
    
    // Time the forward pass
    start_timing(&timings[timing_index], "Forward Pass");
    int *predictions = forward_pass(data);
    end_timing(&timings[timing_index++]);
    
    // Time the accuracy calculation
    start_timing(&timings[timing_index], "Accuracy Calculation");
    // Compare the first 10 predictions with the actual digits.
    printf("\nComparing first 100 predictions with actual digits:\n");
    for (int i = 0; i < 100 && i < data_nrows; i++) {
        printf("Sample %d: Predicted %d, Actual %.0f\n", i, predictions[i], digits[i]);
    }

    double accuracy = final_result(predictions, digits, data_nrows);
    printf("\nFinal Prediction Accuracy: %.2f%%\n", accuracy);
    error_log(predictions, digits, data_nrows, 1000);
    end_timing(&timings[timing_index++]);
    
    // Print timing results
    printf("\n=== Performance Measurements ===\n");
    print_timing_header();
    
    for (int i = 0; i < timing_index; i++) {
        print_timing(&timings[i]);
        total_time += timings[i].cpu_time;
    }
    
    printf("├─────────────────────────────────────┼───────────────┤\n");
    printf("│ Total Time                          │ %11.4f s │\n", total_time);
    print_timing_footer();
    
    // Print final results
    printf("\nFinal Prediction Accuracy: %.2f%%\n", accuracy);
    
    free(predictions);
    unload_data();
    return 0;
}
