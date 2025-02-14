#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <stdio.h>  // file handling functions
#include <stdlib.h> // atoi
#include <string.h> // strtok
#include <sys/wait.h>
#include <errno.h>
#define stacksize 1048576

// descargar biases.csv y weights
    // hecho, biases y wiethgts descargados 
// fgetc para leer de ya abiertos ficheros
// strtok para separar strings

static double **data;
int data_nrows;
int data_ncols = 784;
char *my_path = "/home/penguin/IRCSO/Practica5/files"; // tendréis que poner vuestro path

int seed = 0;
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

int read_matrix(double **mat, char *file, int nrows, int ncols, int fac) {
    /*
     * Dada una matrix (mat), un nombre de fichero (file), una cantidad de filas
     * (nrows) y columnas (ncols), y un multiplicador (fac, no se usa, es 1), deja en mat la
     * matriz (de dimensión nrows x ncols) de datos contenida en el fichero con
     * nombre file
     */

    char *buffer; // Esto contendrá toda la fila
    char *record;  //Esto contendrá las columnas de la fila
    FILE *fstream = fopen(biases3_3.csv, "r");   //file reemplazar por nombre (biases3_3)
    double aux;
    // Hay que hacer control de errores
        int control_errores(){
        FILE *f = fopen(biases3_3.csv, "r");  //file x nombre 
        if(f==NULL){
            printf("errno: %d\n", errno);
            printf("Error: %s\n", strerror(errno));
            perror("Houston, tenemos un problema");
        }
        else
            printf("No tenemos problemas\n");
            
    }
    for (int row = 0; row < nrows; row++) {
    // Leer, separar, y reservar columnas de la fila
        // 
        for (int column = 0; column < ncols; column++) {
            if (record) {
                aux = strtod(record, NULL) * (float)fac;
                mat[row][column] = aux;
            } else {
                mat[row][column] = -1.0;
            }
            // Siguiente Token
            char* token = strtok(str, " ")
        }
    }
    // Hay que cerrar ficheros y liberar memoria
    FILE *f = fclose(biases3_3.csv, “r”); 		//cerrar
     // Unload_data ya lo hace, se puede poner eso y ya.
    unload_data();
    return 0;
}

int read_vector(double *vect, char *file, int nrows) {
    /*
     * Dado un vector (vect), un nombre de fichero (file), y una cantidad de filas
     * (nrows), deja en vect el vector (de dimensión nrows) de datos contenido en
     * el fichero con nombre file
     */
    char *buffer = // Esto contendrá el valor
    FILE *fstream = fopen(weights3_3.csv, "r");
    double aux;
    // Control de errores
    int control_errores(){
        FILE *f = fopen(weights3_3.csv, "r");  //file x nombre 
        if(f==NULL){
            printf("errno: %d\n", errno);
            printf("Error: %s\n", strerror(errno));
            perror("Houston, tenemos un problema");
        }
        else
            printf("No tenemos problemas\n");
            
    }
    for (int row = 0; row < nrows; row++) {
        // leer el valor
        aux = strtod(buffer, NULL);
        vect[row] = aux;
    }
    // Hay que cerrar ficheros y liberar memoria
        // Unload_data ya lo hace, se puede poner eso y ya.
    unload_data();
    FILE *f = fclose(weights3_3.csv, “r”); 		//cerrar pero 

    return 0;
}

void print_matrix(double **mat, int nrows, int ncols, int offset_row,
                  int offset_col) {
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

    digits = malloc(data_nrows *
                    sizeof(double)); // Los valores que idealmente predeciremos
    sprintf(str, "%scsvs/digits.csv", path);
    read_vector(digits, str, data_nrows);

    // Las matrices
    mat1 = malloc(matrices_rows[0] * sizeof(*mat1));
    sprintf(str, "%sparameters/weights%d_%d.csv", path, 0, seed);
    read_matrix(mat1, str, matrices_rows[0], matrices_columns[0], 1);

    // Los vectores
    vec1 = malloc(vector_rows[0] * sizeof(double));
    sprintf(str, "%sparameters/biases%d_%d.csv", path, 0, seed);
    read_vector(vec1, str, vector_rows[0]);

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
    
    unload_data();
    return 0;
}


