#include "descriptores.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include <algorithm>        // Para encontrar máximo en un vector
#include <chrono>           // Medimos tiempo de ejecución
#include <cstdlib>          // Funciones extras, util para terminar procesos
#include <filesystem>       // Manejar sistemas de archivos, sólo en C++17 (cambio en el Makefile también)
#include <iostream>         // Para la entrada y salida
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
                  std::vector<cv::Mat> &imagenTransformada, std::vector<cv::Mat> &imagenNoResize)
{
    // Esta función debe leer el directorio de imagenes que se pasa con
    // el argumento de "carpeta", por el otro lado, el argumento "archivo" tiene una
    // referencia a los nombre de las imagenes

    std::cout << "Se empiezan a leer las imagenes" << std::endl;

    // Iteración de los archivos, adaptación de la función util.py
    for (const auto &entrada : fs::directory_iterator(carpeta))
    {

        if (fs::is_regular_file(entrada.path()) && entrada.path().extension() == ".jpg")
        {
            std::string nombre =
                entrada.path().filename().string(); // Ocupamos .stem() para obtener el nombre sin la extensión
            archivos.push_back(nombre);

            cv::Mat imagen_gris = cv::imread(entrada.path().string(), cv::IMREAD_GRAYSCALE); // Lo mismo pero en gris
            if (imagen_gris.empty())
            {
                std::cerr << "Error en pasar a matriz la imagen: " << nombre << std::endl;
                exit(1); // Se termina todo el proceso
            }
            imagenNoResize.push_back(imagen_gris);
            cv::Mat imagenGrisFourier;
            // Aprovechamos de redimensionar
            cv::resize(imagen_gris, imagenGrisFourier, cv::Size(20, 20), 0, 0, cv::INTER_AREA);

            cv::resize(imagen_gris, imagen_gris, cv::Size(256, 256), 0, 0, cv::INTER_AREA);

            // Dado que ya se está en la iteración, se aprovecha de calcular la DFT
            cv::Mat imagenPushFourier, imagenPushLog;
            imagenGrisFourier.convertTo(imagenPushFourier, CV_32F);
            cv::dft(imagenPushFourier, imagenPushFourier, cv::DFT_COMPLEX_OUTPUT);
            visualizar_magnitud(imagenPushFourier, imagenPushLog);
            // Añadimos al vector de imagen de tipo Mat
            imagenG.push_back(imagen_gris);
            imagenTransformada.push_back(imagenPushLog);
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
    leerImagenes(dir_input_imagenes_R, imagenes, imagenesGrisCV, imagenesTransformada, imagenNoResize);

    // 2. Calcular descriptores de imágenes
    // std::vector<cv::Mat> descriptores;
    std::vector<cv::Mat> descriptoresTextura;
    std::vector<cv::Mat> descriptorIntensidad;
    std::vector<cv::Mat> descriptorBorde;
    std::vector<cv::Mat> descriptorHu;

    std::thread t1(descriptoresGabor, imagenesTransformada, std::ref(descriptoresTextura));
    std::thread t2(histogramaIntensidades, imagenesGrisCV, std::ref(descriptorIntensidad));
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
