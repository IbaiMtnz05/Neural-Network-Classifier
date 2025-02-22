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
int data_nrows = 100; 
int data_ncols = 784;
char *my_path = "/workspaces/Trabajo5/";

int seed = 3;
int matrices_rows[4] = {784, 200, 100, 50};
int matrices_columns[4] = {200, 100, 50, 10};
int vector_rows[4] = {200, 100, 50, 10};
char *str;

static double *digits;
static double **mat1;
static double **mat2;
static double **mat3;
static double **mat4;
static double *vec1;
static double *vec2;
static double *vec3;
static double *vec4;

char *siguiente_token(char *buffer) {
    static char *last_ptr = NULL;
    if (buffer != NULL) {
        last_ptr = buffer;
        return strtok(last_ptr, " ,\n");
    }
    return strtok(NULL, " ,\n");
}

int read_matrix(double **mat, char *file, int nrows, int ncols, int fac) {
    printf("\nLeyendo matriz: %s\n", file);
    
    FILE *fstream = fopen(file, "r");
    if (!fstream || control_errores(file) != 0) {
        return 1;
    }

    char buffer[4096];  
    for (int row = 0; row < nrows; row++) {
        if (fgets(buffer, sizeof(buffer), fstream) == NULL) {
            break;
        }
        char *record = siguiente_token(buffer);

        for (int column = 0; column < ncols; column++) {
            if (record) {
                mat[row][column] = strtod(record, NULL) * fac;
                record = siguiente_token(NULL);
            } else {
                mat[row][column] = -1.0;
            }
        }
    }
    fclose(fstream);
    return 0;
}

int read_vector(double *vect, char *file, int nrows) {
    printf("\nLeyendo vector: %s\n", file);

    FILE *fstream = fopen(file, "r");
    if (!fstream || control_errores(file) != 0) {
        return 1;
    }

    char buffer[256];
    for (int row = 0; row < nrows; row++) {
        if (fgets(buffer, sizeof(buffer), fstream) == NULL) {
            break;
        }
        vect[row] = strtod(buffer, NULL);
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

void load_data(char *path) {
    printf("Cargando digits\n");
    str = malloc(128);
    digits = malloc(data_nrows * sizeof(double));
    sprintf(str, "%scsvs/digits.csv", path);
    read_vector(digits, str, data_nrows);
    printf("Digits cargados\n");

    printf("\nCargando data\n");
    data = malloc(data_nrows * sizeof(double *));
    for (int i = 0; i < data_nrows; i++) {
        data[i] = malloc(data_ncols * sizeof(double));
    }
    sprintf(str, "%scsvs/data.csv", path);
    read_matrix(data, str, data_nrows, data_ncols, 1);
    printf("\nData cargada\n");

    print_matrix(data, 5, 5, 0, 0);

    // Inicialización correcta de las matrices
    mat1 = malloc(matrices_rows[0] * sizeof(double *));
    mat2 = malloc(matrices_rows[1] * sizeof(double *));
    mat3 = malloc(matrices_rows[2] * sizeof(double *));
    mat4 = malloc(matrices_rows[3] * sizeof(double *));
    double **mats[] = {mat1, mat2, mat3, mat4};

    for (int i = 0; i < 4; i++) {
        printf("\nCargando mat%d\n", i + 1);
        for (int j = 0; j < matrices_rows[i]; j++) {
            mats[i][j] = malloc(matrices_columns[i] * sizeof(double));
        }
        sprintf(str, "%sparameters/weights%d_%d.csv", path, i, seed);
        read_matrix(mats[i], str, matrices_rows[i], matrices_columns[i], 1);
        printf("mat%d cargada\n", i + 1);
        print_matrix(mats[i], 5, 5, 0, 0);
    }

    double *vecs[] = {vec1, vec2, vec3, vec4};
    for (int i = 0; i < 4; i++) {
        vecs[i] = malloc(vector_rows[i] * sizeof(double));
        sprintf(str, "%sparameters/biases%d_%d.csv", path, i, seed);
        read_vector(vecs[i], str, vector_rows[i]);
        printf("vec%d cargada\n", i + 1);
    }
}

void unload_data() {
    free(digits);
    for (int i = 0; i < data_nrows; i++) free(data[i]);
    free(data);

    double **mats[] = {mat1, mat2, mat3, mat4};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < matrices_rows[i]; j++) {
            free(mats[i][j]);
        }
        free(mats[i]);
    }

    double *vecs[] = {vec1, vec2, vec3, vec4};
    for (int i = 0; i < 4; i++) free(vecs[i]);

    free(str);
}

int control_errores(const char *checkFile) {
    FILE *f = fopen(checkFile, "r");
    if (!f) {
        printf("errno: %d\n", errno);
        printf("Error: %s\n", strerror(errno));
        perror("Houston, tenemos un problema");
        return 1;
    }
    fclose(f);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("El programa debe recibir un único argumento: la cantidad de procesos que se van a generar.\n");
        return 1;
    }

    load_data(my_path);
    unload_data();
    return 0;
}
