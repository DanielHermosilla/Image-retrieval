#include "descriptores.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/features2d.hpp"
#include "opencv2/objdetect.hpp" // Para el HOG descriptor
#include <algorithm>             // Para encontrar máximo en un vector
#include <chrono>                // Medimos tiempo de ejecución
#include <cmath>
#include <cstdint>
#include <cstdlib>           // Funciones extras, util para terminar procesos
#include <filesystem>        // Manejar sistemas de archivos, sólo en C++17 (cambio en el Makefile también)
#include <fstream>           // Para manipular archivos
#include <iostream>          // Para la entrada y salida
#include <nlohmann/json.hpp> // Para poder guardar los datos en formato JSON
#include <omp.h>
#include <opencv2/core.hpp> // Con esto importamos OpenCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

double global_max = 20;
void visualizar_magnitud(const cv::Mat &frec_complex, cv::Mat &imageLogMagnitud)
{
    // separar complejos en parte real e imaginaria
    cv::Mat frec_real;
    cv::Mat frec_imag;
    frec_real.create(frec_complex.size(), CV_32FC1);
    frec_imag.create(frec_complex.size(), CV_32FC1);
    cv::Mat frec_planos[] = {frec_real, frec_imag};
    cv::split(frec_complex, frec_planos);
    // calcular la magnitud
    cv::Mat magnitud;
    cv::magnitude(frec_real, frec_imag, magnitud);
    // calcular log magnitud para visualizar
    cv::log(magnitud + 1, magnitud);
    double max;
    cv::minMaxIdx(magnitud, NULL, &max);
    if (max > global_max)
    {
        global_max = std::max(max, global_max);
        std::cout << " max=" << global_max << std::endl;
    }
    cv::convertScaleAbs(magnitud, imageLogMagnitud, 255.0 / max, 0);
}
cv::Mat imagenCentrada(const cv::Mat &imageLogMagnitud)
{
    // Centrar la imagen. Código obtenido del material docente del curso
    int w = imageLogMagnitud.cols;
    int h = imageLogMagnitud.rows;
    int w2 = w / 2;
    int h2 = h / 2;
    cv::Rect r00(0, 0, w2, h2);
    cv::Rect r01(w - w2, 0, w2, h2);
    cv::Rect r10(0, h - h2, w2, h2);
    cv::Rect r11(w - w2, h - h2, w2, h2);
    cv::Mat imageCenter(h, w, imageLogMagnitud.type());
    imageLogMagnitud(r00).copyTo(imageCenter(r11));
    imageLogMagnitud(r01).copyTo(imageCenter(r10));
    imageLogMagnitud(r10).copyTo(imageCenter(r01));
    imageLogMagnitud(r11).copyTo(imageCenter(r00));
    return imageCenter;
}

void leerImagenes(const std::string &carpeta, std::vector<std::string> &archivos, std::vector<cv::Mat> &imagenG,
                  std::vector<cv::Mat> &imagenTransformada, std::vector<cv::Mat> &imagenNoResize,
                  std::vector<cv::Mat> &imagenColor)
{
    std::cout << "Se empiezan a leer las imagenes" << std::endl;

    // Step 1: Collect file paths into a vector
    std::vector<fs::path> file_paths;
    for (const auto &entrada : fs::directory_iterator(carpeta))
    {
        if (fs::is_regular_file(entrada.path()) && entrada.path().extension() == ".jpg")
        {
            file_paths.push_back(entrada.path());
        }
    }

// Step 2: Parallel processing with OpenMP
#pragma omp parallel for
    for (size_t i = 0; i < file_paths.size(); ++i)
    {
        const fs::path &file_path = file_paths[i];
        std::string nombre = file_path.filename().string();
        cv::Mat imagen_color = cv::imread(file_path.string(), cv::IMREAD_COLOR);
        cv::Mat imagen_gris = cv::imread(file_path.string(), cv::IMREAD_GRAYSCALE);
        if (imagen_color.empty())
        {
#pragma omp critical
            std::cerr << "Error al cargar la imagen: " << nombre << std::endl;
            continue;
        }

        // Resize and process images
        cv::Mat imagenGrisFourier, imagenPushFourier, imagenPushLog;
        cv::resize(imagen_gris, imagenGrisFourier, cv::Size(20, 20), 0, 0, cv::INTER_AREA);
        cv::resize(imagen_gris, imagen_gris, cv::Size(256, 256), 0, 0, cv::INTER_AREA);
        cv::resize(imagen_color, imagen_color, cv::Size(256, 256), 0, 0, cv::INTER_AREA);

        imagenGrisFourier.convertTo(imagenPushFourier, CV_32F);
        cv::dft(imagenPushFourier, imagenPushFourier, cv::DFT_COMPLEX_OUTPUT);
        visualizar_magnitud(imagenPushFourier, imagenPushLog);

// Store results in shared vectors (order not guaranteed)
#pragma omp critical
        {
            archivos.push_back(nombre);
            imagenNoResize.push_back(imagen_gris);
            imagenG.push_back(imagen_gris);
            imagenTransformada.push_back(imagenPushLog);
            imagenColor.push_back(imagen_color);
        }
    }

    std::cout << "Se terminó de leer las imagenes" << std::endl;
}

void tarea1_indexar(const std::string &dir_input_imagenes_R, const std::string &dir_output_descriptores_R)
{
    if (!fs::exists(dir_input_imagenes_R))
    {
        std::cerr << "ERROR: no existe directorio " << dir_input_imagenes_R << std::endl;
        std::exit(1);
    }
    if (fs::exists(dir_output_descriptores_R))
    {
        std::cerr << "ERROR: ya existe directorio " << dir_output_descriptores_R << std::endl;
        std::exit(1);
    }

    // Crear la carpeta de salida
    fs::create_directories(dir_output_descriptores_R);

    // 1. Leer imágenes en dir_input_imagenes_R

    // Vector con los nombres de las imagenes.jpg
    std::vector<std::string> imagenes;

    // Vector con las imagenes que se leen de tipo Mat
    std::vector<cv::Mat> imagenesGrisCV;
    std::vector<cv::Mat> imagenesTransformada;
    std::vector<cv::Mat> imagenNoResize;
    std::vector<cv::Mat> imagenesColor;
    leerImagenes(dir_input_imagenes_R, imagenes, imagenesGrisCV, imagenesTransformada, imagenNoResize, imagenesColor);

    // 2. Calcular descriptores de imágenes
    // std::vector<cv::Mat> descriptores;
    std::vector<cv::Mat> descriptoresTextura;
    std::vector<cv::Mat> descriptorIntensidad;
    std::vector<cv::Mat> descriptorBorde;
    std::vector<cv::Mat> descriptorHu;

    std::thread t1(descriptoresGabor, imagenesTransformada, std::ref(descriptoresTextura));
    std::thread t2(histogramaIntensidades, imagenesColor, std::ref(descriptorIntensidad));
    std::thread t3(descriptorHOG, imagenesGrisCV, std::ref(descriptorBorde));
    std::thread t4(momentosHu, imagenNoResize, std::ref(descriptorHu));

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    // 3. Escribir descriptores en dir_output_descriptores_R

    std::string archivo_salida = dir_output_descriptores_R + "/resultados.yml";
    cv::FileStorage fs1(archivo_salida, cv::FileStorage::WRITE);

    fs1 << "nombreImagen" << imagenes;
    fs1 << "descriptorTextura" << descriptoresTextura;
    fs1 << "descriptorIntensidad" << descriptorIntensidad;
    fs1 << "descriptorBorde" << descriptorBorde;
    fs1 << "momentosHu" << descriptorHu;

    fs1.release();
    std::cout << "Se genera " << archivo_salida << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Uso: " << argv[0] << " [dir_input_imagenes_R] [dir_output_descriptores_R]" << std::endl;
        return 1;
    }

    std::string dir_input_imagenes_R = argv[1];
    std::string dir_output_descriptores_R = argv[2];

    auto inicio = std::chrono::high_resolution_clock::now();
    tarea1_indexar(dir_input_imagenes_R, dir_output_descriptores_R);
    auto fin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duracion = fin - inicio;
    std::cout << "Tiempo de ejecución del programa de indexación: " << duracion.count() << " segundos" << std::endl;
    return 0;
}
