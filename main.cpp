#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

cv::Mat transparentTemplate(cv::Mat);
cv::Mat sobel(cv::Mat);
void matchTemplateMask(cv::InputArray _img, cv::InputArray _templ, cv::OutputArray _result, int method, cv::InputArray _mask);
void crossCorr(const cv::Mat &img, const cv::Mat &_templ, cv::Mat &corr,
        cv::Size corrsize, int ctype,
        cv::Point anchor, double delta, int borderType);

int main() {
    cv::Mat final, result;
    cv::Mat src = cv::imread("single.png", 0);
    cv::Mat img_template = cv::imread("template.png", 0);
    src.copyTo(final);
    int result_cols = src.cols - img_template.cols + 1;
    int result_rows = src.rows - img_template.rows + 1;

    result.create(result_cols, result_rows, CV_32FC1);
    cv::matchTemplate(src, img_template, result, CV_TM_CCOEFF_NORMED);
    std::cout << "test" << std::endl;
    cv::imwrite("Matched.png", result);
    cv::normalize(result, result, 0, 1, CV_MINMAX, -1, cv::Mat());
    cv::imwrite("Normalized.png", result);

    while(true) {
        double minVal;
        double maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;
        cv::Point matchLoc;

        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
        matchLoc = maxLoc;
        if(maxVal >= 0.3) {
            rectangle(final, matchLoc, cv::Point(matchLoc.x + 50, matchLoc.y + 50), cv::Scalar(0, 0, 255, 255), 2, 4, 0);
            cv::floodFill(result, matchLoc, cv::Scalar(0), 0, cvScalar(0.1), cvScalar(1.));
        } else
            break;
    }

    cv::imwrite("result.png", final);

    return 0;
}

cv::Mat transparentTemplate(cv::Mat raw) {
    cv::Mat alpha, gray, dest;

    cv::cvtColor(raw, gray, CV_RGB2GRAY);
    cv::threshold(gray, alpha, 20, 255, CV_THRESH_BINARY);
    cv::bitwise_not(gray, gray);
    cv::cvtColor(gray, gray, CV_GRAY2BGR);

    /*cv::Mat rgb[3];
    cv::split(raw, rgb);

    cv::Mat rgba[4] = {rgb[0], rgb[1], rgb[2], alpha};

    cv::merge(rgba, 4, dest);*/
    return gray;
}

void matchTemplateMask(cv::InputArray _img, cv::InputArray _templ, cv::OutputArray _result, int method, cv::InputArray _mask) {
    using namespace cv;

    int type = _img.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(CV_TM_SQDIFF <= method && method <= CV_TM_CCOEFF_NORMED);
    Mat img = _img.getMat(), templ = _templ.getMat(), mask = _mask.getMat();int ttype = templ.type(), tdepth = CV_MAT_DEPTH(ttype), tcn = CV_MAT_CN(ttype);
    int mtype = img.type(), mdepth = CV_MAT_DEPTH(type), mcn = CV_MAT_CN(mtype);
    if (depth == CV_8U) {
        depth = CV_32F;
        type = CV_MAKETYPE(CV_32F, cn);
        img.convertTo(img, type, 1.0 / 255);
    }
    if (tdepth == CV_8U) {
        tdepth = CV_32F;
        ttype = CV_MAKETYPE(CV_32F, tcn);
        templ.convertTo(templ, ttype, 1.0 / 255);
    }
    if (mdepth == CV_8U) {
        mdepth = CV_32F;
        mtype = CV_MAKETYPE(CV_32F, mcn);
        compare(mask, Scalar::all(0), mask, CMP_NE);
        mask.convertTo(mask, mtype, 1.0 / 255);
    }
    Size corrSize(img.cols - templ.cols + 1, img.rows - templ.rows + 1);
    _result.create(corrSize, CV_32F);
    Mat result = _result.getMat();
    Mat img2 = img.mul(img);
    Mat mask2 = mask.mul(mask);
    Mat mask_templ = templ.mul(mask);
    Scalar templMean, templSdv;
    double templSum2 = 0;
    meanStdDev(mask_templ, templMean, templSdv);
    templSum2 = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];
    templSum2 += templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];
    templSum2 *= ((double) templ.rows * templ.cols);
    if (method == CV_TM_SQDIFF) {
        Mat mask2_templ = templ.mul(mask2);
        Mat corr(corrSize, CV_32F);
        crossCorr(img, mask2_templ, corr, corr.size(), corr.type(), Point(0, 0), 0, 0);
        crossCorr(img2, mask, result, result.size(), result.type(), Point(0, 0), 0, 0);
        result -= corr * 2;
        result += templSum2;
    }
    else if (method == CV_TM_CCORR_NORMED) {
        if (templSum2 < DBL_EPSILON) {
            result = Scalar::all(1);
            return;
        }
        Mat corr(corrSize, CV_32F);
        crossCorr(img2, mask2, corr, corr.size(), corr.type(), Point(0, 0), 0, 0);
        crossCorr(img, mask_templ, result, result.size(), result.type(), Point(0, 0), 0, 0);
        sqrt(corr, corr);
        result = result.mul(1 / corr);
        result /= std::sqrt(templSum2);
    }
}

void crossCorr(const cv::Mat &img, const cv::Mat &_templ, cv::Mat &corr,
        cv::Size corrsize, int ctype,
        cv::Point anchor, double delta, int borderType) {
    using namespace cv;

    const double blockScale = 4.5;
    const int minBlockSize = 256;
    std::vector<uchar> buf;
    Mat templ = _templ;
    int depth = img.depth(), cn = img.channels();
    int tdepth = templ.depth(), tcn = templ.channels();
    int cdepth = CV_MAT_DEPTH(ctype), ccn = CV_MAT_CN(ctype);
    CV_Assert(img.dims <= 2 && templ.dims <= 2 && corr.dims <= 2);
    if (depth != tdepth && tdepth != std::max(CV_32F, depth)) {
        _templ.convertTo(templ, std::max(CV_32F, depth));
        tdepth = templ.depth();
    }
    CV_Assert(depth == tdepth || tdepth == CV_32F);
    CV_Assert(corrsize.height <= img.rows + templ.rows - 1 &&
            corrsize.width <= img.cols + templ.cols - 1);
    CV_Assert(ccn == 1 || delta == 0);
    corr.create(corrsize, ctype);
    int maxDepth = depth > CV_8S ? CV_64F : std::max(std::max(CV_32F, tdepth), cdepth);
    Size blocksize, dftsize;
    blocksize.width = cvRound(templ.cols * blockScale);
    blocksize.width = std::max(blocksize.width, minBlockSize - templ.cols + 1);
    blocksize.width = std::min(blocksize.width, corr.cols);
    blocksize.height = cvRound(templ.rows * blockScale);
    blocksize.height = std::max(blocksize.height, minBlockSize - templ.rows + 1);
    blocksize.height = std::min(blocksize.height, corr.rows);
    dftsize.width = std::max(getOptimalDFTSize(blocksize.width + templ.cols - 1), 2);
    dftsize.height = getOptimalDFTSize(blocksize.height + templ.rows - 1);
    if (dftsize.width <= 0 || dftsize.height <= 0)
        CV_Error(CV_StsOutOfRange, "the input arrays are too big");
// recompute block size
    blocksize.width = dftsize.width - templ.cols + 1;
    blocksize.width = MIN(blocksize.width, corr.cols);
    blocksize.height = dftsize.height - templ.rows + 1;
    blocksize.height = MIN(blocksize.height, corr.rows);
    Mat dftTempl(dftsize.height * tcn, dftsize.width, maxDepth);
    Mat dftImg(dftsize, maxDepth);
    int i, k, bufSize = 0;
    if (tcn > 1 && tdepth != maxDepth)
        bufSize = templ.cols * templ.rows * CV_ELEM_SIZE(tdepth);
    if (cn > 1 && depth != maxDepth)
        bufSize = std::max(bufSize, (blocksize.width + templ.cols - 1) *
                (blocksize.height + templ.rows - 1) * CV_ELEM_SIZE(depth));
    if ((ccn > 1 || cn > 1) && cdepth != maxDepth)
        bufSize = std::max(bufSize, blocksize.width * blocksize.height * CV_ELEM_SIZE(cdepth));
    buf.resize(bufSize);
// compute DFT of each template plane
    for (k = 0; k < tcn; k++) {
        int yofs = k * dftsize.height;
        Mat src = templ;
        Mat dst(dftTempl, Rect(0, yofs, dftsize.width, dftsize.height));
        Mat dst1(dftTempl, Rect(0, yofs, templ.cols, templ.rows));
        if (tcn > 1) {
            src = tdepth == maxDepth ? dst1 : Mat(templ.size(), tdepth, &buf[0]);
            int pairs[] = {k, 0};
            mixChannels(&templ, 1, &src, 1, pairs, 1);
        }
        if (dst1.data != src.data)
            src.convertTo(dst1, dst1.depth());
        if (dst.cols > templ.cols) {
            Mat part(dst, Range(0, templ.rows), Range(templ.cols, dst.cols));
            part = Scalar::all(0);
        }
        dft(dst, dst, 0, templ.rows);
    }
    int tileCountX = (corr.cols + blocksize.width - 1) / blocksize.width;
    int tileCountY = (corr.rows + blocksize.height - 1) / blocksize.height;
    int tileCount = tileCountX * tileCountY;
    Size wholeSize = img.size();
    Point roiofs(0, 0);
    Mat img0 = img;
    if (!(borderType & BORDER_ISOLATED)) {
        img.locateROI(wholeSize, roiofs);
        img0.adjustROI(roiofs.y, wholeSize.height - img.rows - roiofs.y,
                roiofs.x, wholeSize.width - img.cols - roiofs.x);
    }
    borderType |= BORDER_ISOLATED;
// calculate correlation by blocks
    for (i = 0; i < tileCount; i++) {
        int x = (i % tileCountX) * blocksize.width;
        int y = (i / tileCountX) * blocksize.height;
        Size bsz(std::min(blocksize.width, corr.cols - x),
                std::min(blocksize.height, corr.rows - y));
        Size dsz(bsz.width + templ.cols - 1, bsz.height + templ.rows - 1);
        int x0 = x - anchor.x + roiofs.x, y0 = y - anchor.y + roiofs.y;
        int x1 = std::max(0, x0), y1 = std::max(0, y0);
        int x2 = std::min(img0.cols, x0 + dsz.width);
        int y2 = std::min(img0.rows, y0 + dsz.height);
        Mat src0(img0, Range(y1, y2), Range(x1, x2));
        Mat dst(dftImg, Rect(0, 0, dsz.width, dsz.height));
        Mat dst1(dftImg, Rect(x1 - x0, y1 - y0, x2 - x1, y2 - y1));
        Mat cdst(corr, Rect(x, y, bsz.width, bsz.height));
        for (k = 0; k < cn; k++) {
            Mat src = src0;
            dftImg = Scalar::all(0);
            if (cn > 1) {
                src = depth == maxDepth ? dst1 : Mat(y2 - y1, x2 - x1, depth, &buf[0]);
                int pairs[] = {k, 0};
                mixChannels(&src0, 1, &src, 1, pairs, 1);
            }
            if (dst1.data != src.data)
                src.convertTo(dst1, dst1.depth());
            if (x2 - x1 < dsz.width || y2 - y1 < dsz.height)
                copyMakeBorder(dst1, dst, y1 - y0, dst.rows - dst1.rows - (y1 - y0),
                        x1 - x0, dst.cols - dst1.cols - (x1 - x0), borderType);
            dft(dftImg, dftImg, 0, dsz.height);
            Mat dftTempl1(dftTempl, Rect(0, tcn > 1 ? k * dftsize.height : 0,
                    dftsize.width, dftsize.height));
            mulSpectrums(dftImg, dftTempl1, dftImg, 0, true);
            dft(dftImg, dftImg, DFT_INVERSE + DFT_SCALE, bsz.height);
            src = dftImg(Rect(0, 0, bsz.width, bsz.height));
            if (ccn > 1) {
                if (cdepth != maxDepth) {
                    Mat plane(bsz, cdepth, &buf[0]);
                    src.convertTo(plane, cdepth, 1, delta);
                    src = plane;
                }
                int pairs[] = {0, k};
                mixChannels(&src, 1, &cdst, 1, pairs, 1);
            }
            else {
                if (k == 0)
                    src.convertTo(cdst, cdepth, 1, delta);
                else {
                    if (maxDepth != cdepth) {
                        Mat plane(bsz, cdepth, &buf[0]);
                        src.convertTo(plane, cdepth);
                        src = plane;
                    }
                    add(src, cdst, cdst);
                }
            }
        }
    }
}
