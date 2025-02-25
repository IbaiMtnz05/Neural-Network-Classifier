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

# Compilación
all: $(SRC)
	gcc -o $(TARGET) $(SRC) $(LDFLAGS)

# Limpieza de archivos compilados
clean:
	rm -f $(TARGET)
