#include <iostream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char ** argv)
{
    cv::Mat image;
    image=cv::imread("./imageBasics/dog.jpg");

    if(image.data==nullptr)
    {
        cerr<<"文件"<<argv[1]<<"不存在"<<endl;
        return 0;
    }

    cout<<"图像宽为"<<image.cols<<"，高为"<<image.rows<<"，通道数为"<<image.channels()<<endl;
    cv::imshow("image",image);
    cv::waitKey(0); //暂停程序，等待按键输入

    if(image.type()!=CV_8UC1 && image.type() != CV_8UC3)
    {
        cout<< "请输入一张彩色图或灰度图" << endl;
        return 0;
    }
    //cout<<"data: "<< cv::format(image,cv::Formatter::FMT_PYTHON)<<std::endl;

    //遍历图像
    for(auto y=0;y<image.rows;y++)
    {
        unsigned char *row_ptr=image.ptr<unsigned char>(y);
        for(auto x=0;x<image.cols;x++)
        {
            auto data_ptr= &row_ptr[x*image.channels()];

            for(int c=0;c != image.channels();c++)
            {
                unsigned char data=data_ptr[c];
            }
        }
    }

    cv::Mat image_another = image; //直接赋值不会拷贝数据
    image_another(cv::Rect(0,0,100,100)).setTo(0);
    cv::imshow("modify image", image);
    cv::waitKey(0);//延迟为0

    cv::Mat image_clone;
    image.copyTo(image_clone);
    image_clone(cv::Rect(0,0,100,100)).setTo(100);
    cv::imshow("image", image);
    cv::imshow("cloned_image",image_clone);
    cv::waitKey(0);

    //关闭窗口
    cv::destroyAllWindows();

    return 0;
}