# Nombre del ejecutable
TARGET = main

# Archivos fuente
SRC = main.c

# Librerías
LDFLAGS = -lSDL2

# Instalación de dependencias
install:
	sudo apt update
	sudo apt install -y libsdl2-dev
	wget https://github.com/IbaiMtnz05/Trabajo5/raw/refs/heads/main/csvs/data.csv
	mv data.csv csvs/

# Compilación
all: $(SRC)
	gcc -o $(TARGET) $(SRC) $(LDFLAGS)

# Limpieza de archivos compilados
clean:
	rm -f $(TARGET)
