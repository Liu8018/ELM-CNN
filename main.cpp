#include <iostream>
#include "ELM_CNN_Model.h"

int main()
{
    cv::Mat A =(cv::Mat_<float>(5,5)<< 0,-1, 0,-1, 5,
                                       -1, 0,-1, 0,-1,
                                        0,-1, 5,-1, 0,
                                       -1, 0,-1, 0,-1,
                                        5,-1, 0,-1, 0);
    
    std::cout<<"src:\n"<<A<<std::endl;
    
    std::vector<cv::Mat> mats;
    mats.push_back(A);
    
    cv::Mat B;
    deconvInputMats(mats,5,5,2,1,B);
    
    std::cout<<"deconved:\n"<<B<<std::endl;
    
    return 0;
}
