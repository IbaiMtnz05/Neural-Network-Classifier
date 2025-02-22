#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <stdio.h>  // file handling functions
#include <stdlib.h> // atoi
#include <string.h> // strtok
#include <sys/wait.h>
#include <errno.h> //control de errores
#define stacksize 1048576

// Forward declarations of all functions
int control_errores(const char *checkFile);
int read_matrix(double **mat, char *file, int nrows, int ncols, int fac);
int read_vector(double *vect, char *file, int nrows);
void print_matrix(double **mat, int nrows, int ncols, int offset_row, int offset_col);
void load_data(char *path);
void unload_data(void);
double** mat_mul(double** input, double** weights, int index);
double** sum_vect(double** matrix, double* vector, int index);
double** relu(double** matrix, int index);
int* argmax(double** matrix, int rows, int cols);
void free_matrix(double** matrix, int rows);
int* forward_pass(double** data);
char *siguiente_token(char *buffer);

// Global variables
static double **data;
int data_nrows;
int data_ncols = 784;
char *my_path = "/workspaces/Trabajo5/";

int seed = 3;
int matrices_rows[4] = {784, 200, 100, 50};
int matrices_columns[4] = {200, 100, 50, 10};
int vector_rows[4] = {200, 100, 50, 10};
char *str;
int rows_per_div;

static double *digits;
static double **mat1;
static double **mat2;
static double **mat3;
static double **mat4;
static double *vec1;
static double *vec2;
static double *vec3;
static double *vec4;

// Función global para tokenizar la línea
char *siguiente_token(char *buffer) {
    static char *last_ptr = NULL;
    if (buffer != NULL) {
        last_ptr = buffer;
        return strtok(last_ptr, " ,\n");
    } else {
        return strtok(NULL, " ,\n");
    }
}

int read_matrix(double **mat, char *file, int nrows, int ncols, int fac) {
    /*
     * Dada una matrix (mat), un nombre de fichero (file), una cantidad de filas
     * (nrows) y columnas (ncols), y un multiplicador (fac, no se usa, es 1), deja en mat la
     * matriz (de dimensión nrows x ncols) de datos contenida en el fichero con
     * nombre file
     */
    printf("\nRead matrix\n");
    char *buffer = malloc(data_nrows * sizeof(double)); // Esto contendrá toda la fila
    char *record;  //Esto contendrá las columnas de la fila
    FILE *fstream = fopen(file, "r");
    double aux;
    // Hay que hacer control de errores
    if (control_errores(file) != 0) {
        return 1; // Error opening file
    }

    for (int row = 0; row < nrows; row++) {
	// Leer, separar, y reservar columnas de la fila
        for (int column = 0; column < ncols; column++) {
            if (record) {
                aux = strtod(record, NULL) * (float)fac;
                mat[row][column] = aux;
            } else {
                mat[row][column] = -1.0;
            }
            record = siguiente_token(buffer);
            // Siguiente Token
        }
    }
    // Hay que cerrar ficheros y liberar memoria
    printf("cerrando archivos y liberando memoria\n");
    fclose(fstream);
    printf("haciendo free buffer");
    free(buffer);
    return 0;
}

int read_vector(double *vect, char *file, int nrows) {
    /*
     * Dado un vector (vect), un nombre de fichero (file), y una cantidad de filas
     * (nrows), deja en vect el vector (de dimensión nrows) de datos contenido en
     * el fichero con nombre file
     */
    printf("\nRead vector\n");
    char *buffer = malloc(nrows*sizeof(double)); // Esto contendrá el valor
    FILE *fstream = fopen(file, "r");
    double aux;
    // Control de errores
        if (control_errores(file) != 0) {
        return 1; // Error opening file
    }

    for (int row = 0; row < nrows; row++) {
        // leer el valor
        aux = strtod(buffer, NULL);
        vect[row] = aux;
    }
    // Hay que cerrar ficheros y liberar memoria
    printf("Cerrando archivos y liberando memoria \n");
    fclose(fstream);
    free(buffer);
    return 0;
}


void print_matrix(double **mat, int nrows, int ncols, int offset_row, int offset_col) {
    /*
     * Dada una matriz (mat), una cantidad de filas (nrows) y columnas (ncols) a
     * imprimir, y una cantidad de filas (offset_row) y columnas (offset_col) a
     * ignorar, imprime por salida estándar nrows x ncols de la matriz
     */
    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            printf("%f ", mat[row + offset_row][col + offset_col]);
        }
        printf("\n");
    }
}


void load_data(char *path) {

    /*
     * Dado un directorio en el que están los datos y parámetros, los carga en las
     * variables de entorno
     */
    printf("cargando digits\n");
    str = malloc(128);
    digits = malloc(data_nrows * sizeof(double)); // Los valores que idealmente predeciremos
    sprintf(str, "%scsvs/digits.csv", path);
    read_vector(digits, str, data_nrows);
    printf("digits cargados\n");

    printf("\ncargando data\n");
    data = malloc(data_ncols * data_nrows * sizeof(double));
    sprintf(str, "%scsvs/data.csv", path);
    read_matrix(data, str, data_nrows, data_ncols, 1);
    printf("\ndata cargada\n");
  
    // Las matrices
    printf("\ncargando mat1");
    mat1 = malloc(matrices_rows[0] * sizeof(*mat1));
    mat1 = malloc(matrices_rows[0] * sizeof(double *));
    for (int i = 0; i < matrices_rows[0]; i++) {
        mat1[i] = malloc(matrices_columns[0] * sizeof(double));
    }

    sprintf(str, "%sparameters/weights%d_%d.csv", path, 0, seed);
    read_matrix(mat1, str, matrices_rows[0], matrices_columns[0], 1);
    printf("\nmat1 cargada\n");

    printf("cargando mat2\n");
    mat2 = malloc(matrices_rows[1] * sizeof(*mat2));
    mat2 = malloc(matrices_rows[1] * sizeof(double *));
    for (int i = 0; i < matrices_rows[1]; i++) {
        mat2[i] = malloc(matrices_columns[1] * sizeof(double));
    }
    sprintf(str, "%sparameters/weights%d_%d.csv", path, 1, seed);
    read_matrix(mat1, str, matrices_rows[1], matrices_columns[1], 1);
    printf("\nmat2 cargada\n");

    printf("cargando mat3\n");
    mat3 = malloc(matrices_rows[2] * sizeof(*mat3));
    mat3 = malloc(matrices_rows[2] * sizeof(double *));
    for (int i = 0; i < matrices_rows[2]; i++) {
        mat3[i] = malloc(matrices_columns[2] * sizeof(double));
    }
    sprintf(str, "%sparameters/weights%d_%d.csv", path, 2, seed);
    read_matrix(mat1, str, matrices_rows[2], matrices_columns[2], 1);
    printf("\nmat3 cargada\n");
    
    printf("cargando mat4\n");
    mat4 = malloc(matrices_rows[3] * sizeof(*mat4));
    mat4 = malloc(matrices_rows[3] * sizeof(double *));
    for (int i = 0; i < matrices_rows[0]; i++) {
        mat4[i] = malloc(matrices_columns[3] * sizeof(double));
    }
    sprintf(str, "%sparameters/weights%d_%d.csv", path, 3, seed);
    read_matrix(mat1, str, matrices_rows[3], matrices_columns[3], 1);
    printf("\nmat4 cargada\n");

    // Los vectores
    vec1 = malloc(vector_rows[0] * sizeof(double));
    sprintf(str, "%sparameters/biases%d_%d.csv", path, 0, seed);
    read_vector(vec1, str, vector_rows[0]);
    printf("vec1 cargada\n");

    vec2 = malloc(vector_rows[1] * sizeof(double));
    sprintf(str, "%sparameters/biases%d_%d.csv", path, 1, seed);
    read_vector(vec1, str, vector_rows[1]);
    printf("vec2 cargada\n");

    vec3 = malloc(vector_rows[2] * sizeof(double));
    sprintf(str, "%sparameters/biases%d_%d.csv", path, 2, seed);
    read_vector(vec1, str, vector_rows[2]);
    printf("vec3 cargada\n");

    vec4 = malloc(vector_rows[3] * sizeof(double));
    sprintf(str, "%sparameters/biases%d_%d.csv", path, 3, seed);
    read_vector(vec1, str, vector_rows[3]);
    printf("vec4 cargada\n");
}

void unload_data() {
    /*
     * Liberar la memoria
     */
    free(digits);
    free(data);
    free(mat1);
    free(mat2);
    free(mat3);
    free(mat4);
    free(vec1);
    free(vec2);
    free(vec3);
    free(vec4);
    free(str);
}



void print(void *arg) { printf("Hola, soy %d\n", *(int *)arg); }

int main(int argc, char *argv[]) {
    /*
     * El programa recibe un único argumento, la cantidad de procesos que se
     * emplearán en la paralelización. Por ejemplo, parallel 3 tendrá que dividir
     * la matriz en tres, y lanzar tres procesos paralelos. Cada proceso, deberá
     * procesar un tercio de la matriz de datos
     */

    if (argc != 2) {
        printf("El programa debe tener un único argumento, la cantidad de procesos que se van a generar\n");
        exit(1);
    }

    load_data(my_path);
    int* predictions = forward_pass(data);
    // If you want to compare with actual digits:
    printf("\nComparing first 10 predictions with actual digits:\n");
    for(int i = 0; i < 10 && i < data_nrows; i++) {
        printf("Sample %d: Predicted %d, Actual %.0f\n", i, predictions[i], digits[i]);
    }
    free(predictions);
    unload_data();
    return 0;
}

int control_errores(const char *checkFile) {
    FILE *f = fopen(checkFile, "r");
    
    if (f == NULL) {
        printf("errno: %d\n", errno);
        printf("Error: %s\n", strerror(errno));
        perror("Houston, tenemos un problema");
        return 1; // Indica error
    }
    
    printf("No tenemos problemas con el archivo: %s\n", checkFile);
    fclose(f); // Cerrar el archivo si se abrió correctamente
    return 0; // Indica que no hubo error
}

// Matrix multiplication function (already discussed)
double** mat_mul(double** input, double** weights, int index) {
    int rows = matrices_rows[index];
    int cols = matrices_columns[index];
    int input_cols = matrices_columns[index-1];
    
    double** result = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        result[i] = malloc(cols * sizeof(double));
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = 0;
            for (int k = 0; k < input_cols; k++) {
                result[i][j] += input[i][k] * weights[k][j];
            }
        }
    }
    
    return result;
}

// Add bias vector to each row of the matrix
double** sum_vect(double** matrix, double* vector, int index) {
    int rows = matrices_rows[index];
    int cols = matrices_columns[index];
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] += vector[j];
        }
    }
    
    return matrix;
}

// ReLU activation function: max(0, x)
double** relu(double** matrix, int index) {
    int rows = matrices_rows[index];
    int cols = matrices_columns[index];
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix[i][j] < 0) {
                matrix[i][j] = 0;
            }
        }
    }
    
    return matrix;
}

// ArgMax function - returns index of maximum value in each row
int* argmax(double** matrix, int rows, int cols) {
    int* predictions = malloc(rows * sizeof(int));
    
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

// Helper function to free a matrix
void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int* forward_pass(double** data) {
    double** capa0, **capa1, **capa2, **capa3, **resultado;
    int* predicciones;
    
    printf("\n=== Starting Forward Pass ===\n");
    
    // Layer 0
    printf("\n--- Layer 0 ---\n");
    printf("Performing matrix multiplication (data * mat1)...\n");
    capa0 = mat_mul(data, mat1, 0);
    printf("Adding bias vector 1...\n");
    capa0 = sum_vect(capa0, vec1, 0);
    printf("Applying ReLU activation...\n");
    capa0 = relu(capa0, 0);
    printf("Layer 0 complete. Output shape: [%d x %d]\n", matrices_rows[0], matrices_columns[0]);
    
    // Layer 1
    printf("\n--- Layer 1 ---\n");
    printf("Performing matrix multiplication (capa0 * mat2)...\n");
    capa1 = mat_mul(capa0, mat2, 1);
    printf("Adding bias vector 2...\n");
    capa1 = sum_vect(capa1, vec2, 1);
    printf("Applying ReLU activation...\n");
    capa1 = relu(capa1, 1);
    printf("Layer 1 complete. Output shape: [%d x %d]\n", matrices_rows[1], matrices_columns[1]);
    free_matrix(capa0, matrices_rows[0]);
    printf("Freed layer 0 memory\n");
    
    // Layer 2
    printf("\n--- Layer 2 ---\n");
    printf("Performing matrix multiplication (capa1 * mat3)...\n");
    capa2 = mat_mul(capa1, mat3, 2);
    printf("Adding bias vector 3...\n");
    capa2 = sum_vect(capa2, vec3, 2);
    printf("Applying ReLU activation...\n");
    capa2 = relu(capa2, 2);
    printf("Layer 2 complete. Output shape: [%d x %d]\n", matrices_rows[2], matrices_columns[2]);
    free_matrix(capa1, matrices_rows[1]);
    printf("Freed layer 1 memory\n");
    
    // Layer 3
    printf("\n--- Layer 3 (Final Layer) ---\n");
    printf("Performing matrix multiplication (capa2 * mat4)...\n");
    capa3 = mat_mul(capa2, mat4, 3);
    printf("Adding bias vector 4...\n");
    capa3 = sum_vect(capa3, vec4, 3);
    printf("Applying ReLU activation...\n");
    resultado = relu(capa3, 3);
    printf("Layer 3 complete. Output shape: [%d x %d]\n", matrices_rows[3], matrices_columns[3]);
    free_matrix(capa2, matrices_rows[2]);
    printf("Freed layer 2 memory\n");
    
    // Get predictions
    printf("\n--- Computing Final Predictions ---\n");
    printf("Calculating argmax for each row...\n");
    predicciones = argmax(resultado, data_nrows, 10);
    printf("Predictions computed for %d samples\n", data_nrows);
    
    // Print first few predictions
    printf("\nFirst 10 predictions:\n");
    for(int i = 0; i < 10 && i < data_nrows; i++) {
        printf("Sample %d: Predicted digit %d\n", i, predicciones[i]);
    }
    
    free_matrix(resultado, matrices_rows[3]);
    printf("Freed final layer memory\n");
    
    printf("\n=== Forward Pass Complete ===\n");
    
    return predicciones;
}