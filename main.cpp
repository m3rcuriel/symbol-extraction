#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <iostream>
#include <cmath>
#include <array>

void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void parseInput(int argc, char* argv[], int*, int*, std::string*);
struct PASS {
    cv::Point* one = 0;
    cv::Point* two = 0;
    cv::Mat* display;
};

int main(int argc, char* argv[]) {
    int min, max;
    std::string file;
    parseInput(argc, argv, &min, &max, &file);

    cv::Mat input = cv::imread(file, 0);
    cv::Mat display;
    cv::cvtColor(input, display, cv::COLOR_GRAY2RGB);

    if(argc == 2) {
        cv::namedWindow("Test", cv::WINDOW_NORMAL);

        struct PASS data;
        data.display = &display;

        cv::setMouseCallback("Test", CallBackFunc, &data);


        do {
            cv::imshow("Test", display);
            cv::waitKey(30);
        } while(!data.one || !data.two);

        cv::destroyAllWindows();

        double dx = data.two->x - data.one->x;
        double dy = data.two->y - data.one->y;
        double radius = sqrt(dx * dx + dy * dy) / 2;

        min = radius - 7;
        max = radius + 7;
    }

    cv::Mat transform = input.clone();
    cv::cvtColor(input, input, cv::COLOR_GRAY2RGB);
    std::vector<cv::Vec3f> circles;
    cv::GaussianBlur(transform, transform, cv::Size(9,9), 1);
    cv::Canny(transform, transform, 3, 70, 5);
    cv::HoughCircles(transform, circles, CV_HOUGH_GRADIENT, 1, 30, 70, 27, min, max);

    for( size_t i = 0; i < circles.size(); i++ )
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        cv::circle( input, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0 );
        // circle outline
        cv::circle( input, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0 );
    }
    cv::imwrite("result.png", input);
}

void CallBackFunc(int event, int x, int y, int flags __attribute__((unused)), void* userdata) {
    struct PASS* data = static_cast<PASS*>(userdata);
    if( event == cv::EVENT_LBUTTONDOWN ) {
        if(data->one)
            data->two = new cv::Point(x, y);
        else
            data->one = new cv::Point(x, y);
    }
}

void parseInput(int argc, char* argv[], int* min, int* max, std::string* file) {
    if(argc < 2) {
        std::cerr << "You must supply a filename to run this program" << std::endl;
        exit(-1);
    } else if (argc > 3) {
        std::cerr << "Too many arguments!" << std::endl;
        exit(-1);
    }
    if(argc == 2) {
        *min = 20;
        *max = 30;
        *file = argv[1];
    } else {
        *min = atoi(argv[2]) - 5;
        *max = atoi(argv[2]) + 5;
        *file = argv[1];
    }
}
