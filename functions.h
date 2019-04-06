#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <dirent.h>
#include <algorithm>

//输入图像去卷积化
void deconvInputMats(const std::vector<cv::Mat> &inputMats, 
                     int Fh, int Fw, int padding, int stride,
                     cv::Mat &deconvedMat);

//展开邻域像素
void neighborPix2Line(const cv::Mat &img, int x, int y, int Fw, int Fh, cv::Mat &line);

#endif // FUNCTIONS_H
