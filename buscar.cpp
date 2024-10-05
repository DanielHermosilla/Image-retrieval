#include "descriptores.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include <algorithm>  // Para encontrar máximo en un vector
#include <chrono>     // Medimos tiempo de ejecución
#include <cstdlib>    // Funciones extras, util para terminar procesos
#include <cstdlib>    // Para system()
#include <filesystem> // Manejar sistemas de archivos, sólo en C++17 (cambio en el Makefile también)
#include <fstream>    // Para manipular archivos
#include <iostream>   // Para la entrada y salida
#include <numeric>
#include <omp.h>
#include <opencv2/core.hpp> // Con esto importamos OpenCV
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp> // Para SURF
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

            // Aprovechamos de redimensionar
            imagenNoResize.push_back(imagen_gris);
            cv::Mat imagenGrisFourier;
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

void leerYML(const std::string &directorio, std::vector<std::string> &nombres, std::vector<cv::Mat> &descriptorTextura,
             std::vector<cv::Mat> &descriptorIntensidad, std::vector<cv::Mat> &descriptorBorde,
             std::vector<cv::Mat> &descriptorHu)
{
    for (const auto &entrada : fs::directory_iterator(directorio))
    {
        std::string nombreArchivo = entrada.path().string();
        std::cout << "Se está leyendo el archivo " << nombreArchivo << std::endl;

        cv::FileStorage fs2(nombreArchivo, cv::FileStorage::READ);

        fs2["nombreImagen"] >> nombres;
        fs2["descriptorTextura"] >> descriptorTextura;
        fs2["descriptorIntensidad"] >> descriptorIntensidad;
        fs2["descriptorBorde"] >> descriptorBorde;
        fs2["momentosHu"] >> descriptorHu;
        fs2.release();
        std::cout << "Se leyó correctamente " << nombreArchivo << std::endl;
    }
}
// Función para obtener los índices ordenados por las distancias
// Al final no se ocupó, aunque a futuro se podría proponer un sistema de ranking.
std::vector<int> ranking(const std::vector<float> &distancia)
{
    // Vector para los índices
    std::vector<int> indices(distancia.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Esto devolverá un vector con punteros que muestran dónde se ubica el menor a mayor elemento
    std::stable_sort(indices.begin(), indices.end(),
                     [&distancia](int i1, int i2) { return distancia[i1] < distancia[i2]; });
    // Crear el vector de ranking
    std::vector<int> posiciones(distancia.size(), 0);
    for (size_t i = 0; i < indices.size(); i++)
    {
        posiciones[indices[i]] = i;
    }

    return posiciones;
}
void leer_imagenes(const std::string &carpeta_entrada, const std::string &carpeta_descriptores,
                   const std::string &output_file)
{
    std::vector<cv::Mat> descriptorTexturaR;
    std::vector<cv::Mat> descriptorIntensidadR;
    std::vector<cv::Mat> descriptorBordeR;
    std::vector<cv::Mat> descriptorHuR;

    std::vector<cv::Mat> descriptorTexturaQ;
    std::vector<cv::Mat> descriptorIntensidadQ;
    std::vector<cv::Mat> descriptorBordeQ;
    std::vector<cv::Mat> descriptorHuQ;
    std::vector<std::string> nombresQ;
    std::vector<std::string> nombresR;
    std::vector<cv::Mat> imagenesQ;
    std::vector<cv::Mat> imagenesQTransformada;
    std::vector<cv::Mat> imagenesNoResize;

    leerYML(carpeta_descriptores, nombresR, descriptorTexturaR, descriptorIntensidadR, descriptorBordeR, descriptorHuR);
    std::cout << "Se terminó de leer los descriptores de las imágenes calculadas" << std::endl;
    std::cout << "Se empezará a crear los descriptores de la carpeta de entrada" << std::endl;
    leerImagenes(carpeta_entrada, nombresQ, imagenesQ, imagenesQTransformada, imagenesNoResize);
    // Se corren en paralelo, son procesos independientes
    std::thread t1(descriptoresGabor, imagenesQTransformada, std::ref(descriptorTexturaQ));
    std::thread t2(histogramaIntensidades, imagenesQ, std::ref(descriptorIntensidadQ));
    std::thread t3(descriptorHOG, imagenesQ, std::ref(descriptorBordeQ));
    std::thread t4(momentosHu, imagenesNoResize, std::ref(descriptorHuQ));
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    std::cout << "Número de descriptores en Q: "
              << descriptorTexturaQ.size() + descriptorBordeQ.size() + descriptorHuQ.size() +
                     descriptorIntensidadQ.size()
              << std::endl;
    std::cout << "Número de descriptores en R: "
              << descriptorTexturaR.size() + descriptorBordeR.size() + descriptorHuR.size() +
                     descriptorIntensidadR.size()
              << std::endl;
    std::cout << "Se empieza a crear el archivo con los resultados" << std::endl;
    std::ofstream outfile(output_file);

    int contador = 0;
    size_t i;
#pragma omp parallel for private(i)
    for (i = 0; i < descriptorTexturaQ.size(); i++)
    {
        std::vector<float> distanciasTexturas;
        std::vector<float> distanciasIntensidad;
        std::vector<float> distanciasBorde;
        std::vector<float> distanciasHu;

        for (size_t k = 0; k < descriptorTexturaR.size(); k++)
        {
            float distB = cv::norm(descriptorBordeR[k], descriptorBordeQ[i], cv::NORM_L2);
            float distT = cv::norm(descriptorTexturaR[k], descriptorTexturaQ[i], cv::NORM_L2);
            float distI = cv::norm(descriptorIntensidadR[k], descriptorIntensidadQ[i], cv::NORM_L2);
            float distH = cv::norm(descriptorHuQ[i], descriptorHuR[k], cv::NORM_L2);
            distanciasTexturas.push_back(distT);
            distanciasIntensidad.push_back(distI);
            distanciasBorde.push_back(distB);
            distanciasHu.push_back(distH);
        }

        auto min_indiceTextura = std::min_element(distanciasTexturas.begin(), distanciasTexturas.end());
        float min_valorTextura = *min_indiceTextura;
        size_t min_indiceTextura2 = std::distance(distanciasTexturas.begin(), min_indiceTextura);

        auto min_indiceHu = std::min_element(distanciasHu.begin(), distanciasHu.end());
        float min_valorHu = *min_indiceHu;
        size_t min_indiceHu2 = std::distance(distanciasHu.begin(), min_indiceHu);

        auto min_indiceIntensidad = std::min_element(distanciasIntensidad.begin(), distanciasIntensidad.end());
        float min_valorIntensidad = *min_indiceIntensidad;
        size_t min_indiceIntensidad2 = std::distance(distanciasIntensidad.begin(), min_indiceIntensidad);

        auto min_indiceBorde = std::min_element(distanciasBorde.begin(), distanciasBorde.end());
        float min_valorBorde = *min_indiceBorde;
        size_t min_indiceBorde2 = std::distance(distanciasBorde.begin(), min_indiceBorde);
#pragma omp critical
        {
            if (min_valorTextura <= 70)
            {
                // Precisión es de aprox 371/377 = 98.4%
                outfile << nombresQ[i] << "\t" << nombresR[min_indiceTextura2] << "\t" << min_valorTextura << std::endl;
            }
            else if (min_valorBorde <= 5800)
            {
                // Precisión es de aprox 637/645= 98,6%
                outfile << nombresQ[i] << "\t" << nombresR[min_indiceBorde2] << "\t" << min_valorBorde << std::endl;
            }
            else if (min_valorHu <= 65000)
            {
                // Precisión es de aprox 44/46 = 95.6%
                outfile << nombresQ[i] << "\t" << nombresR[min_indiceHu2] << "\t" << min_valorHu << std::endl;
            }

            else
            {
                outfile << nombresQ[i] << "\t" << nombresR[min_indiceIntensidad2] << "\t" << min_valorIntensidad
                        << std::endl;
            }

            contador++;
        }
        if (contador % 500 == 0)
        {
            std::cout << "Procesados " << contador << " matches..." << std::endl;
        }
    }

    outfile.close();
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Uso: " << argv[0] << " <carpeta de entrada> <carpeta de descriptores> <archivo de salida>"
                  << std::endl;
        return 1;
    }

    std::string carpeta_entrada = argv[1];
    std::string carpeta_descriptores = argv[2];
    std::string output_file = argv[3];

    auto inicio = std::chrono::high_resolution_clock::now();
    leer_imagenes(carpeta_entrada, carpeta_descriptores, output_file);
    auto fin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duracion = fin - inicio;
    std::cout << "Tiempo de ejecución del programa de indexación: " << duracion.count() << " segundos" << std::endl;
    return 0;
}
