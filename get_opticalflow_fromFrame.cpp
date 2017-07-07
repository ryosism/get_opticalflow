// compile
// g++ `pkg-config --libs opencv` `pkg-config --cflags opencv` get_opticalflow.cpp

#include <opencv2/opencv.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <string>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <iomanip>

using namespace std;
using namespace cv;
using namespace cv::superres;

int main()
{
        // TV-L1アルゴリズムによるオプティカルフロー計算オブジェクトの生成
        Ptr<DenseOpticalFlowExt> opticalFlow = superres::createOptFlow_DualTVL1();

        int frame = 1;
        while (1)
        {
            waitKey(1000);
            std::stringstream prevFilename;
            prevFilename << "0002/" << frame << ".jpg";
            std::string prevfn = prevFilename.str();
            Mat prev = cv::imread(prevfn.c_str());
            cv::imshow("prevFrame", prev);
            printf("prev = %s\n", prevfn.c_str());

            std::stringstream currFilename;
            currFilename << "0002/" << frame+2 << ".jpg";
            std::string currfn = currFilename.str();
            Mat curr = cv::imread(currfn.c_str());
            cv::imshow("currFrame", curr);
            printf("curr = %s\n", currfn.c_str());

            // オプティカルフローの計算
            Mat flowX, flowY;
            opticalFlow->calc(prev, curr, flowX, flowY);
            // printf("dims = %d\n", flowX.dims);
            // printf("rows = %d\n", flowX.rows);
            // std::cout << "flowY=" << flowY << std::endl << std::endl;

            // オプティカルフローの可視化（色符号化）
            //  オプティカルフローを極座標に変換（角度は[deg]）
            Mat magnitude, angle;
            cartToPolar(flowX, flowY, magnitude, angle, true);
            //  色相（H）はオプティカルフローの角度
            //  彩度（S）は0～1に正規化したオプティカルフローの大きさ
            //  明度（V）は1
            Mat hsvPlanes[3];
            hsvPlanes[0] = angle;
            normalize(magnitude, magnitude, 0, 1, NORM_MINMAX); // 正規化
            hsvPlanes[1] = magnitude;
            hsvPlanes[2] = Mat::ones(magnitude.size(), CV_32F);
            //  HSVを合成して一枚の画像にする
            Mat hsv;
            merge(hsvPlanes, 3, hsv);
            //  HSVからBGRに変換
            Mat flowBgr;
            cvtColor(hsv, flowBgr, cv::COLOR_HSV2BGR);

            // 表示
            cv::imshow("optical flow", flowBgr);
	        std::stringstream ss;
            ss << setfill('0') << setw(5) << right << frame << ".jpg";
            std::string fn = ss.str();
            flowBgr.convertTo(flowBgr, CV_8U, 255);
       	    printf("%s\n",fn.c_str());
            cv::imwrite(fn,flowBgr);

            frame += 1;
        }
}
