#ifndef descriptores
#define descriptores
#include "opencv2/core/mat.hpp"

void histogramaIntensidades(const std::vector<cv::Mat> &imagen, std::vector<cv::Mat> &descriptorIntensidad);
void descriptorHOG(const std::vector<cv::Mat> &imagenes, std::vector<cv::Mat> &descriptorBorde);
void momentosHu(const std::vector<cv::Mat> &imagenes, std::vector<cv::Mat> &descriptoresHu);
void descriptoresGabor(const std::vector<cv::Mat> &imagenes, std::vector<cv::Mat> &descriptoresTextura);

#endif // descriptores
