#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

// Estructura para matrices dinámicas
typedef struct {
    int rows;
    int cols;
    double **data;
} DynamicMatrix;

// Estructura para vectores dinámicos
typedef struct {
    int size;
    double *data;
} DynamicVector;

// Funciones para matrices dinámicas
DynamicMatrix* createMatrix(int rows, int cols) {
    DynamicMatrix* matrix = malloc(sizeof(DynamicMatrix));
    matrix->rows = rows;
    matrix->cols = cols;
    
    // Asignar memoria para filas
    matrix->data = malloc(rows * sizeof(double*));
    
    // Asignar memoria para columnas en cada fila
    for (int i = 0; i < rows; i++) {
        matrix->data[i] = malloc(cols * sizeof(double));
        // Inicializar a cero (opcional)
        memset(matrix->data[i], 0, cols * sizeof(double));
    }
    
    return matrix;
}

void freeMatrix(DynamicMatrix* matrix) {
    if (!matrix) return;
    
    // Liberar cada fila
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->data[i]);
    }
    
    // Liberar el array de punteros a filas
    free(matrix->data);
    
    // Liberar la estructura
    free(matrix);
}

// Funciones para vectores dinámicos
DynamicVector* createVector(int size) {
    DynamicVector* vector = malloc(sizeof(DynamicVector));
    vector->size = size;
    vector->data = malloc(size * sizeof(double));
    
    // Inicializar a cero (opcional)
    memset(vector->data, 0, size * sizeof(double));
    
    return vector;
}

void freeVector(DynamicVector* vector) {
    if (!vector) return;
    free(vector->data);
    free(vector);
}

// Función para leer una matriz desde un archivo
int readMatrix(DynamicMatrix* matrix, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error al abrir el archivo %s\n", filename);
        return -1;
    }
    
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    
    for (int row = 0; row < matrix->rows; row++) {
        if ((read = getline(&line, &len, file)) == -1) {
            fprintf(stderr, "Error: archivo con menos filas de las esperadas\n");
            free(line);
            fclose(file);
            return -1;
        }
        
        char* token = strtok(line, " ,\n");
        for (int col = 0; col < matrix->cols; col++) {
            if (token) {
                matrix->data[row][col] = strtod(token, NULL);
                token = strtok(NULL, " ,\n");
            } else {
                matrix->data[row][col] = 0.0;
            }
        }
    }
    
    free(line);
    fclose(file);
    return 0;
}

// Función para leer un vector desde un archivo
int readVector(DynamicVector* vector, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error al abrir el archivo %s\n", filename);
        return -1;
    }
    
    char* line = NULL;
    size_t len = 0;
    
    for (int i = 0; i < vector->size; i++) {
        if (getline(&line, &len, file) != -1) {
            vector->data[i] = strtod(line, NULL);
        } else {
            vector->data[i] = 0.0;
        }
    }
    
    free(line);
    fclose(file);
    return 0;
}

// Función para imprimir una matriz
void printMatrix(DynamicMatrix* matrix, int max_rows, int max_cols) {
    int rows = (max_rows > 0 && max_rows < matrix->rows) ? max_rows : matrix->rows;
    int cols = (max_cols > 0 && max_cols < matrix->cols) ? max_cols : matrix->cols;
    
    printf("Matriz %dx%d (mostrando %dx%d):\n", matrix->rows, matrix->cols, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

// Función para imprimir un vector
void printVector(DynamicVector* vector, int max_size) {
    int size = (max_size > 0 && max_size < vector->size) ? max_size : vector->size;
    
    printf("Vector de tamaño %d (mostrando %d elementos):\n", vector->size, size);
    for (int i = 0; i < size; i++) {
        printf("%8.4f ", vector->data[i]);
        if ((i + 1) % 10 == 0) printf("\n");
    }
    printf("\n");
}

// Ejemplo de uso en la función principal
int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Uso: %s <num_procesos>\n", argv[0]);
        return 1;
    }
    
    int num_procesos = atoi(argv[1]);
    if (num_procesos <= 0) {
        printf("El número de procesos debe ser positivo\n");
        return 1;
    }
    
    // Definir dimensiones
    int data_nrows = 60000;  // Número de imágenes
    int data_ncols = 784;    // 28x28 pixels
    
    // Crear la matriz de datos
    DynamicMatrix* data = createMatrix(data_nrows, data_ncols);
    
    // Crear los parámetros del modelo
    int matrices_rows[4] = {784, 200, 100, 50};
    int matrices_columns[4] = {200, 100, 50, 10};
    
    // Crear matrices de pesos
    DynamicMatrix* weights[4];
    for (int i = 0; i < 4; i++) {
        weights[i] = createMatrix(matrices_rows[i], matrices_columns[i]);
    }
    
    // Crear vectores de bias
    DynamicVector* biases[4];
    for (int i = 0; i < 4; i++) {
        biases[i] = createVector(matrices_columns[i]);
    }
    
    // Vector para almacenar los dígitos reales
    DynamicVector* digits = createVector(data_nrows);
    
    // TODO: Cargar datos desde archivos
    char path[256] = "/workspaces/Trabajo5";  // Ajustar según sea necesario
    char filename[512];
    
    // Ejemplo de carga (ajustar rutas según tu estructura)
    sprintf(filename, "%s/csvs/data.csv", path);
    if (readMatrix(data, filename) == 0) {
        printMatrix(data, 60000, 784);  // Mostrar solo primeras 2 filas, 20 columnas
    }
    
    // Liberar memoria
    freeVector(digits);
    for (int i = 0; i < 4; i++) {
        freeMatrix(weights[i]);
        freeVector(biases[i]);
    }
    freeMatrix(data);
    
    return 0;
}