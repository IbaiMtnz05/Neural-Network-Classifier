# Nombre del ejecutable
TARGET = main

# Archivos fuente
SRC = main.c

# Librerías
LDFLAGS = -lSDL2

# Flags de compilación
CFLAGS = -O2 -D_GNU_SOURCE

# Instalación de dependencias
install:
	sudo apt update
	sudo apt install -y libsdl2-dev
	wget https://github.com/IbaiMtnz05/Trabajo5/raw/refs/heads/main/csvs/data.csv
	mv data.csv csvs/

# Compilación
all: $(SRC)
	gcc $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Limpieza de archivos compilados
clean:
	rm -f $(TARGET)
