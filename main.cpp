#include <iostream>
#include "ELM_CNN_Model.h"

int main()
{
    /*
    cv::Mat S =(cv::Mat_<float>(5,5)<<0,1,0,1,5,
                                     1,0,1,0,1,
                                     0,1,5,1,0,
                                     1,0,1,0,1,
                                     5,1,0,1,0);
    cv::Mat A;
    S.convertTo(A,CV_32F);
    */
    
    //std::cout<<"src:\n"<<A<<std::endl;
    
    cv::Mat A = cv::imread("/media/liu/D/linux-windows/图片/Pictures/fagvsfv.jpg");
    cv::resize(A,A,cv::Size(20,20));
    cv::imshow("src",A);
    
    std::vector<cv::Mat> mats;
    mats.push_back(A);
    
    cv::Mat B;
    deconvInputMats(mats,5,5,2,1,B);
    
    B.convertTo(B,CV_8U);
    cv::namedWindow("result",0);
    cv::imshow("result",B);
    cv::waitKey();
    
    //std::cout<<"deconved:\n"<<B<<std::endl;
    
    return 0;
}
