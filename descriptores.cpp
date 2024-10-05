#include "descriptores.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "opencv2/objdetect.hpp" // Para el HOG descriptor
#include <cstdlib>               // Funciones extras, util para terminar procesos
#include <cstdlib>               // Para system()
#include <omp.h>
#include <opencv2/core.hpp> // Con esto importamos OpenCV
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

void descriptoresGabor(const std::vector<cv::Mat> &imagenes, std::vector<cv::Mat> &descriptoresTextura)
{
    // El siguiente paper es de utilidad:
    // http://ijns.jalaxy.com.tw/contents/ijns-v20-n4/ijns-2018-v20-n4-p609-616.pdf
    // 1. Convolución Gabor con cuatro escalas, 8 orientaciones = 32 filtros
    // 2. Las escalas se controlan por lambda, las orientaciones con theta
    // 3. Hay que dividir la imagen en 16 regiones (bloques 64x64)
    // 4. El resultado es, 32x32=1024
    // Voy a iterar sobre lambda ya que esto permite ver distintas escalas de frecuencia. Notar que estamos en imagen
    // logarítmica
    std::vector<float> valoresLambda = {0.1, 0.5, 0.8, 1, 5};

    std::vector<float> valoresTheta = {0,
                                       CV_PI / 16,
                                       CV_PI / 8,
                                       3 * CV_PI / 16,
                                       CV_PI / 8,
                                       CV_PI * 5 / 16,
                                       3 * CV_PI / 8,
                                       7 * CV_PI / 16,
                                       CV_PI / 2,
                                       9 * CV_PI / 16,
                                       5 * CV_PI / 8,
                                       11 * CV_PI / 16,
                                       2 * CV_PI / 3,
                                       13 * CV_PI / 16,
                                       7 * CV_PI / 8,
                                       CV_PI};

    // Primero calcularé el Filtro Gabor para no repetir el cálculo
    std::vector<cv::Mat> filtrosGabor;

    for (size_t l = 0; l < 5; l++)
    {
        for (size_t a = 0; a < 16; a++)
        {
            // Parámetros del filtro de Gabor
            cv::Size ksize = cv::Size(15, 15); // Tamaño del kernel
            double sigma = 1.0;                // Desviacion estandar de la gaussiana
            float lambd = valoresLambda[l];    // Longitud de onda
            double gamma = 1;                  // Relacion de aspecto espacial
            float theta = valoresTheta[a];     // Desfase, 45 grados
            int ktype = CV_32F;                // Tipo de kernel

            cv::Mat kernelTemp = cv::getGaborKernel(ksize, sigma, theta, lambd, gamma, 0.0, ktype);
            filtrosGabor.push_back(kernelTemp);
        }
    }
    descriptoresTextura.resize(imagenes.size());
    size_t k;
#pragma omp parallel for private(k)

    for (size_t k = 0; k < imagenes.size(); k++)
    {
        cv::Mat descriptorImagen;
        cv::Mat imagenActual = imagenes[k];
        for (size_t x = 0; x < filtrosGabor.size(); x++)
        {
            cv::Mat imagenFiltrada;
            cv::filter2D(imagenActual, imagenFiltrada, CV_32F, filtrosGabor[x]);

            for (int ancho = 0; ancho < imagenFiltrada.cols; ancho += imagenFiltrada.cols / 4)
            {
                for (int largo = 0; largo < imagenFiltrada.rows; largo += imagenFiltrada.rows / 4)
                {
                    cv::Rect roi(ancho, largo, imagenFiltrada.cols / 4, imagenFiltrada.rows / 4);
                    cv::Mat imagenCrop = imagenFiltrada(roi);
                    cv::Scalar mean, stddev;
                    cv::meanStdDev(imagenCrop, mean, stddev);
                    descriptorImagen.push_back(static_cast<float>(mean[0]));
                    descriptorImagen.push_back(static_cast<float>(stddev[0]));
                }
            }
        }
        cv::normalize(descriptorImagen, descriptorImagen, 255, 0, cv::NORM_MINMAX);
        descriptoresTextura[k] = descriptorImagen;
    }
}

void momentosHu(const std::vector<cv::Mat> &imagenes, std::vector<cv::Mat> &descriptoresHu)
{
    size_t i;
    descriptoresHu.resize(imagenes.size());
#pragma omp parallel for private(i)
    for (i = 0; i < imagenes.size(); i++)
    {
        cv::Mat descriptorFinal;
        cv::Mat imagenActual = imagenes[i];
        cv::resize(imagenActual, imagenActual, cv::Size(20, 20), 0, 0, cv::INTER_AREA);

        cv::normalize(imagenActual, imagenActual, 255, 0, cv::NORM_MINMAX);
        for (size_t h = 0; h < 1; h++)
        {
            for (int y = 0; y < imagenActual.cols; y += imagenActual.cols / 4)
            {
                for (int x = 0; x < imagenActual.rows; x += imagenActual.rows / 4)
                {
                    cv::Rect bloque(x, y, imagenActual.cols / 4, imagenActual.rows / 4);
                    cv::Mat bloqueImagen = imagenActual(bloque);
                    cv::Moments momentos = cv::moments(bloqueImagen, false);
                    double momentosHu[7];
                    cv::HuMoments(momentos, momentosHu);
                    cv::Mat resultadoMatriz(1, 7, CV_32F);
                    for (int j = 0; j < 7; j++)
                    {
                        resultadoMatriz.at<float>(0, j) = static_cast<float>(momentosHu[j]);
                    }
                    cv::normalize(resultadoMatriz, resultadoMatriz, 10000000000, 0, cv::NORM_MINMAX);
                    descriptorFinal.push_back(resultadoMatriz);
                }
            }
            cv::pyrDown(imagenActual, imagenActual);
        }
        descriptoresHu[i] = descriptorFinal;
    }
}

void histogramaIntensidades(const std::vector<cv::Mat> &imagen, std::vector<cv::Mat> &descriptorIntensidad)
{

    int nbins = 16;
    float rango[] = {0, 255};
    const float *rangoHistograma = rango;

    descriptorIntensidad.resize(imagen.size());
    size_t i;
#pragma omp parallel for private(i)
    for (i = 0; i < imagen.size(); i++)
    {
        cv::Mat imagenActual = imagen[i];
        cv::Mat descriptorImagen;
        for (size_t h = 0; h < 6; h++)
        {
            for (int y = 0; y < imagenActual.cols; y += imagenActual.cols / 4)
            {
                for (int x = 0; x < imagenActual.rows; x += imagenActual.rows / 4)
                {
                    cv::Rect bloque(x, y, imagenActual.cols / 4, imagenActual.rows / 4);
                    cv::Mat bloqueImagen = imagenActual(bloque);
                    cv::Mat histograma;
                    cv::calcHist(&bloqueImagen, 1, 0, cv::Mat(), histograma, 1, &nbins, &rangoHistograma);
                    cv::normalize(histograma, histograma, 255, 0, cv::NORM_MINMAX);
                    histograma.reshape(1, histograma.cols);
                    descriptorImagen.push_back(histograma);
                }
            }
            cv::pyrDown(imagenActual, imagenActual);
        }

        descriptorIntensidad[i] = descriptorImagen;
    }
}

void descriptorHOG(const std::vector<cv::Mat> &imagenes, std::vector<cv::Mat> &descriptorBorde)
{
    // De la documentación de HOGDescriptor nos definimos los parámetros para utilizar una división de tamaño de 4x4 y
    // 16 bins.
    descriptorBorde.resize(imagenes.size());
    size_t i;
#pragma omp parallel for private(i)

    for (i = 0; i < imagenes.size(); i++)
    {
        cv::Mat imagenActual = imagenes[i];
        cv::Mat descriptorImagen;

        for (size_t h = 0; h < 6; h++)
        {
            cv::Size winSize(imagenActual.cols / 4, imagenActual.cols / 4); // Tamaño de la ventana de detección
            cv::Size blockSize(imagenActual.cols / 4,
                               imagenActual.cols / 4); // Tamaño de los bloques en los que se divide la ventana
            cv::Size blockStride(imagenActual.cols / 4, imagenActual.cols / 4); // El desplazamiento entre bloques
            cv::Size cellSize(imagenActual.cols / 4, imagenActual.cols / 4);    // Tamaño de la celda
            int nbins = 9;                                                      // Número de bins

            cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
            for (int y = 0; y < imagenActual.cols; y += imagenActual.cols / 4)
            {
                for (int x = 0; x < imagenActual.rows; x += imagenActual.rows / 4)
                {
                    cv::Rect bloque(x, y, imagenActual.cols / 4, imagenActual.rows / 4);
                    cv::Mat bloqueImagen = imagenActual(bloque);
                    cv::Mat histograma;
                    std::vector<float> temp;
                    hog.compute(imagenActual, temp);
                    cv::Mat tempCopia(1, temp.size(), CV_32F, temp.data());
                    cv::normalize(tempCopia, tempCopia, 255, 0, cv::NORM_MINMAX);
                    descriptorImagen.push_back(tempCopia);
                }
            }
            cv::pyrDown(imagenActual, imagenActual);
        }
        descriptorBorde[i] = descriptorImagen;
    }
}