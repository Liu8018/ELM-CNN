#include "functions.h"

void deconvInputMats(const std::vector<cv::Mat> &inputMats, 
                     int Fh, int Fw, int padding, int stride,
                     cv::Mat &deconvedMat)
{
    if(inputMats.empty())
        return;
    
    //确定一些矩阵大小尺寸数据
    int h = inputMats[0].rows;
    int w = inputMats[0].cols;
    int c = inputMats[0].channels();
    int n = inputMats.size();
    int snXh = ((h+2*padding-Fh)/stride+1)*((w+2*padding-Fw)/stride+1);
    int scXw = Fh*Fw;
    int Xh = snXh*n;
    int Xw = scXw*c;
    
    //给去卷积后的矩阵分配空间
    deconvedMat.create(Xh,Xw,CV_32F);
    
    //去卷积
    for(int i=0;i<n;i++)
    {
        cv::Mat nROI = deconvedMat(cv::Range(i*snXh,(i+1)*snXh),cv::Range(0,Xw));
        
        std::vector<cv::Mat> imgChannels;
        cv::split(inputMats[i],imgChannels);
        
        for(int j=0;j<c;j++)
        {
            cv::Mat X(h+2*padding,w+2*padding,CV_32F,cv::Scalar(0));
            imgChannels[j].copyTo(X(cv::Range(padding,X.rows-padding),cv::Range(padding,X.cols-padding)));
            
            cv::Mat cROI = nROI(cv::Range(0,snXh),cv::Range(j*scXw,(j+1)*scXw));
            
            int xBegin = Fw/2;
            int xEnd = X.cols-Fw/2;
            int yBegin = Fh/2;
            int yEnd = X.rows-Fh/2;
            int rowsId = 0;
            for(int y=yBegin;y<yEnd;y+=stride)
                for(int x=xBegin;x<xEnd;x+=stride)
                {
                    cv::Mat lineROI = cROI(cv::Range(rowsId,rowsId+1),cv::Range(0,cROI.cols));
                    neighborPix2Line(X,x,y,Fw,Fh,lineROI);
                    rowsId++;
                }
        }
        
    }
}

void neighborPix2Line(const cv::Mat &img, int x, int y, int Fw, int Fh, cv::Mat &line)
{
    int xl = x-Fw/2;
    int xh = x+Fw/2;
    int yl = y-Fh/2;
    int yh = y+Fh/2;
    
    int colsId = 0;
    for(int y=yl;y<=yh;y++)
        for(int x=xl;x<=xh;x++)
        {
            line.at<float>(0,colsId) = img.at<float>(y,x);
            colsId++;
        }
    
    /*
    //test
    std::cout<<"neighborPix2Line:-------------------------------------------begin"<<std::endl;
    std::cout<<"input mat:\n"<<img<<std::endl;
    std::cout<<"x:"<<x<<" y:"<<y<<std::endl;
    std::cout<<"result line:\n"<<line<<std::endl;
    std::cout<<"neighborPix2Line:-------------------------------------------end"<<std::endl;
    */
}
