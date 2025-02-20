#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <errno.h>
#define stacksize 1048576

static double **data;
int data_nrows = 60000; // Fixed number of rows for data.csv
int data_ncols = 784;
char *my_path = "/workspaces/Trabajo5";

int seed = 0;
int matrices_rows[4] = {784, 200, 100, 50};
int matrices_columns[4] = {200, 100, 50, 10};
int vector_rows[4] = {200, 100, 50, 10};

static double *digits;
static double **mat1;
static double **mat2;
static double **mat3;
static double **mat4;
static double *vec1;
static double *vec2;
static double *vec3;
static double *vec4;

int read_matrix(double **mat, char *file, int nrows, int ncols, int fac);
int read_vector(double *vect, char *file, int nrows);
void print_matrix(double **mat, int nrows, int ncols, int offset_row, int offset_col);
void load_data(char *path);
void unload_data();
int control_errores(const char *checkFile);

int read_matrix(double **mat, char *file, int nrows, int ncols, int fac) {
    FILE *fstream = fopen(file, "r");
    if (fstream == NULL) {
        perror("Error opening file");
        return -1;
    }

    char buffer[1024];
    int row = 0;
    while (row < nrows && fgets(buffer, sizeof(buffer), fstream) != NULL) {
        buffer[strcspn(buffer, "\n")] = '\0'; // Remove newline
        char *token = strtok(buffer, ",");
        int col = 0;
        while (token != NULL && col < ncols) {
            mat[row][col] = strtod(token, NULL) * fac;
            token = strtok(NULL, ",");
            col++;
        }
        if (col != ncols) {
            fprintf(stderr, "Error: Invalid columns in row %d of %s\n", row, file);
            fclose(fstream);
            return -1;
        }
        row++;
    }
    fclose(fstream);
    return 0;
}

int read_vector(double *vect, char *file, int nrows) {
    FILE *fstream = fopen(file, "r");
    if (fstream == NULL) {
        perror("Error opening file");
        return -1;
    }

    char buffer[1024];
    int row = 0;
    while (row < nrows && fgets(buffer, sizeof(buffer), fstream) != NULL) {
        buffer[strcspn(buffer, "\n")] = '\0';
        vect[row] = strtod(buffer, NULL);
        row++;
    }
    fclose(fstream);
    return 0;
}

void load_data(char *path) {
    char *str = malloc(256 * sizeof(char));
    if (str == NULL) {
        perror("Failed to allocate path buffer");
        exit(EXIT_FAILURE);
    }

    // Load data matrix (data.csv)
    data = malloc(data_nrows * sizeof(double *));
    for (int i = 0; i < data_nrows; i++) {
        data[i] = malloc(data_ncols * sizeof(double));
    }
    snprintf(str, 256, "%s/csvs/data.csv", path);
    read_matrix(data, str, data_nrows, data_ncols, 1);

    // Load digits vector (digits.csv)
    digits = malloc(data_nrows * sizeof(double));
    snprintf(str, 256, "%s/csvs/digits.csv", path);
    read_vector(digits, str, data_nrows);

    // Load layer 0 weights and biases
    mat1 = malloc(matrices_rows[0] * sizeof(double *));
    for (int i = 0; i < matrices_rows[0]; i++) {
        mat1[i] = malloc(matrices_columns[0] * sizeof(double));
    }
    snprintf(str, 256, "%s/parameters/weights0_%d.csv", path, seed);
    read_matrix(mat1, str, matrices_rows[0], matrices_columns[0], 1);

    vec1 = malloc(vector_rows[0] * sizeof(double));
    snprintf(str, 256, "%s/parameters/biases0_%d.csv", path, seed);
    read_vector(vec1, str, vector_rows[0]);

    // Load layer 1 weights and biases
    mat2 = malloc(matrices_rows[1] * sizeof(double *));
    for (int i = 0; i < matrices_rows[1]; i++) {
        mat2[i] = malloc(matrices_columns[1] * sizeof(double));
    }
    snprintf(str, 256, "%s/parameters/weights1_%d.csv", path, seed);
    read_matrix(mat2, str, matrices_rows[1], matrices_columns[1], 1);

    vec2 = malloc(vector_rows[1] * sizeof(double));
    snprintf(str, 256, "%s/parameters/biases1_%d.csv", path, seed);
    read_vector(vec2, str, vector_rows[1]);

    // Load layer 2 weights and biases
    mat3 = malloc(matrices_rows[2] * sizeof(double *));
    for (int i = 0; i < matrices_rows[2]; i++) {
        mat3[i] = malloc(matrices_columns[2] * sizeof(double));
    }
    snprintf(str, 256, "%s/parameters/weights2_%d.csv", path, seed);
    read_matrix(mat3, str, matrices_rows[2], matrices_columns[2], 1);

    vec3 = malloc(vector_rows[2] * sizeof(double));
    snprintf(str, 256, "%s/parameters/biases2_%d.csv", path, seed);
    read_vector(vec3, str, vector_rows[2]);

    // Load layer 3 weights and biases
    mat4 = malloc(matrices_rows[3] * sizeof(double *));
    for (int i = 0; i < matrices_rows[3]; i++) {
        mat4[i] = malloc(matrices_columns[3] * sizeof(double));
    }
    snprintf(str, 256, "%s/parameters/weights3_%d.csv", path, seed);
    read_matrix(mat4, str, matrices_rows[3], matrices_columns[3], 1);

    vec4 = malloc(vector_rows[3] * sizeof(double));
    snprintf(str, 256, "%s/parameters/biases3_%d.csv", path, seed);
    read_vector(vec4, str, vector_rows[3]);

    free(str);
}

void unload_data() {
    free(digits);
    for (int i = 0; i < data_nrows; i++) free(data[i]);
    free(data);

    for (int i = 0; i < matrices_rows[0]; i++) free(mat1[i]);
    free(mat1);
    for (int i = 0; i < matrices_rows[1]; i++) free(mat2[i]);
    free(mat2);
    for (int i = 0; i < matrices_rows[2]; i++) free(mat3[i]);
    free(mat3);
    for (int i = 0; i < matrices_rows[3]; i++) free(mat4[i]);
    free(mat4);

    free(vec1);
    free(vec2);
    free(vec3);
    free(vec4);
}

int control_errores(const char *checkFile) {
    FILE *f = fopen(checkFile, "r");
    if (f == NULL) {
        perror("Error al abrir el archivo");
        return 1;
    }
    fclose(f);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <num_procesos>\n", argv[0]);
        exit(1);
    }

    load_data(my_path);
    // Ejemplo de impresión para verificación
    printf("Primeros 5 elementos de la primera fila de data:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", data[0][i]);
    }
    printf("\n");

    unload_data();
    return 0;
}