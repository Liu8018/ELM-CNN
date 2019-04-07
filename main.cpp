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
    
    /*
    cv::Mat A = cv::imread("/media/liu/D/linux-windows/图片/Pictures/fagvsfv.jpg");
    cv::resize(A,A,cv::Size(20,20));
    cv::namedWindow("src",0);
    cv::imshow("src",A);
    
    std::vector<cv::Mat> mats;
    mats.push_back(A);
    
    cv::Mat B;
    deconvInputMats(mats,5,5,2,1,B);
    
    B.convertTo(B,CV_8U);
    cv::namedWindow("result",0);
    cv::imshow("result",B);
    cv::waitKey();
    */
    
    //std::cout<<"deconved:\n"<<B<<std::endl;
    
    //载入训练和测试数据
    std::vector<cv::Mat> trainImgs;
    std::vector<cv::Mat> testImgs;
    std::vector<std::vector<bool>> trainLabelBins;
    std::vector<std::vector<bool>> testLabelBins;
    loadMnistData_csv("/media/liu/D/linux-windows/dataset/MNIST_data2/mnist_test.csv",
                      0.8,trainImgs,testImgs,trainLabelBins,testLabelBins);
    
    //训练数据去卷积
    int Fw=5,Fh=5,padding=2,stride=1;
    cv::Mat deconvedMat;
    deconvInputMats(trainImgs,Fh,Fw,padding,stride,deconvedMat);
    
    //归一化
    normalize_img(deconvedMat);
    
    //设定卷积核数目
    int k = 3;
    
    //随机生成矩阵
    cv::Mat rw1;
    randomGenerate(rw1,cv::Size(k,deconvedMat.cols));
    
    //计算卷积核
    cv::Mat tH = deconvedMat*rw1;
    sigmoid(tH);
    cv::Mat w2 = tH.inv(1)*deconvedMat;
    cv::Mat F = w2.t();
    
    //计算卷积结果
    cv::Mat H1 = deconvedMat*F;
    
    std::cout<<"test1"<<std::endl;
    //展开
    cv::Mat H1_l;
    reshapeConvedMat(H1,trainImgs.size(),H1_l);
    
    std::cout<<"test2"<<std::endl;
    //随机生成矩阵
    int nHiddenNodes=200;
    cv::Mat rw2;
    randomGenerate(rw2,cv::Size(nHiddenNodes,H1_l.cols));
    
    std::cout<<"test3"<<std::endl;
    //计算全连接层权重
    cv::Mat tH2 = H1_l*rw2;
    sigmoid(tH2);
    cv::Mat trainTarget;
    label2target(trainLabelBins,trainTarget);
    std::cout<<"tH2.size:"<<tH2.size()<<std::endl;
    std::cout<<"trainTarget.size:"<<trainTarget.size()<<std::endl;
    cv::Mat W_FC = (tH2.t()*tH2).inv(1)*tH2.t()*trainTarget;
    
    std::cout<<"test4"<<std::endl;
    //测试
    float score0 = calcScore(tH2*W_FC,trainTarget);
    std::cout<<"score on training data:"<<score0<<std::endl;
    
    
    cv::Mat testDeconvedMat;
    deconvInputMats(testImgs,Fh,Fw,padding,stride,testDeconvedMat);
    
    normalize_img(testDeconvedMat);
    
    std::cout<<"test5"<<std::endl;
    cv::Mat t1 = testDeconvedMat*F;
    cv::Mat t2;
    reshapeConvedMat(t1,testImgs.size(),t2);
    
    std::cout<<"test6"<<std::endl;
    t2 *= rw2;
    sigmoid(t2);
    t2 *= W_FC;
    
    std::cout<<"test7"<<std::endl;
    cv::Mat testTarget;
    label2target(testLabelBins,testTarget);
    float score = calcScore(t2,testTarget);
    
    std::cout<<"score:"<<score<<std::endl;
    
    
    return 0;
}
