# Compilador
CXX := clang++

# Parámetros para compilación: usar C++ 17 y con todos los warnings
CFLAGS += -std=c++17 -Wall -Wextra

# Compilar con optimizaciones. Para debug poner -O0 -ggdb
CFLAGS += -O3

# Incluir directorio actual para headers
CFLAGS += -I.

# Usar OpenCV para compilación y linkeo
CFLAGS += $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs opencv4)

# Añadir flags para OpenMP
CFLAGS += -Xpreprocessor -fopenmp
LDFLAGS += -Xpreprocessor -fopenmp -lomp
LDFLAGS += -pthread


########## árchivos compilar ##########
MAIN_CPP := indexar.cpp 

########## Archivos a generar ##########
MAIN_BIN := bin/$(basename $(MAIN_CPP))

########## Reglas de compilación ##########
# Reglas all y clean no corresponden a archivos
.PHONY: all clean

# No borrar archivos intermedios
.PRECIOUS: build/%.o

# Por defecto se generan todos los ejecutables de los ejemplos
all: $(MAIN_BIN)

# Para cada ejecutable se requiere el object correspondiente más los helpers
$(MAIN_BIN): build/$(basename $(MAIN_CPP)).o build/descriptores.o
	mkdir -p "$(@D)"
	$(CXX) $^ -o $@ $(LDFLAGS)

# Para generar un object se usa el fuente cpp correspondiente más los headers
build/$(basename $(MAIN_CPP)).o: $(MAIN_CPP)
	mkdir -p "$(@D)"
	$(CXX) -c $(CFLAGS) -o $@ $<

build/descriptores.o: descriptores.cpp descriptores.h
	mkdir -p "$(@D)"
	$(CXX) -c $(CFLAGS) -o $@ $<

# Limpiar todos los archivos de compilación
clean: rm -rf bin/ build/
