/*
 * ImageQualityMetric.cpp
 *
 *  Created on: Apr 30, 2014
 *      Author: dailos
 */

#include "ImageQualityMetric.h"
#include "ToolBox.h"

ImageQualityMetric::ImageQualityMetric()
{
  // TODO Auto-generated constructor stub

}

ImageQualityMetric::~ImageQualityMetric()
{
  // TODO Auto-generated destructor stub
}

double ImageQualityMetric::meanSquareError(const cv::Mat& x, const cv::Mat& y)
{ //better use cv::matchTemplate(a, b, result, CV_TM_SQDIFF); result/a.total();
  cv::Mat x_y = x - y;
  return cv::sum(x_y.mul(x_y)).val[0]/x_y.total();
}

double ImageQualityMetric::correlationCoefficient(const cv::Mat& x, const cv::Mat& y)
{ // better use cv::matchTemplate(x, y, result, CV_TM_COEFF_NORMED);
  cv::Scalar meanX, stdDevX, meanY, stdDevY;
  cv::meanStdDev(x, meanX, stdDevX);
  cv::meanStdDev(y, meanY, stdDevY);
  cv::Mat x_meanX = x-meanX;
  cv::Mat y_meanY = y-meanY;
  //cv::multiply(x-meanX, y-meanY, num);
  return cv::sum(x_meanX.mul(y_meanY)).val[0] / std::sqrt(cv::sum(x_meanX.mul(x_meanX)).val[0] * cv::sum(y_meanY.mul(y_meanY)).val[0]);
}

double ImageQualityMetric::covariance(const cv::Mat& x, const cv::Mat& y)
{// better use cv::matchTemplate(a, b, result, CV_TM_CCOEFF); result/a.total();
  cv::Scalar meanX, stdDevX, meanY, stdDevY;
  cv::meanStdDev(x, meanX, stdDevX);
  cv::meanStdDev(y, meanY, stdDevY);
  cv::Mat x_meanX = x-meanX;
  cv::Mat y_meanY = y-meanY;

  return cv::sum(x_meanX.mul(y_meanY)).val[0]/(x_meanX.total());
}

//structural similarity index metric
double ImageQualityMetric::ssim(const cv::Mat& x, const cv::Mat& y)
{
  double l, c, s; //luminance, contrast, structure
  double C1(300), C2(30);  //trivial values to ensure stable solution when denominator is close to zero
  cv::Scalar meanX, stdDevX, meanY, stdDevY;
  cv::meanStdDev(x, meanX, stdDevX);
  cv::meanStdDev(y, meanY, stdDevY);

  l = ((2 * meanX.val[0] * meanY.val[0]) + C1)/(std::pow(meanX.val[0],2) + std::pow(meanY.val[0],2) + C1);
  c = ((2 * stdDevX.val[0] * stdDevY.val[0]) + C2)/(std::pow(stdDevX.val[0],2) + std::pow(stdDevY.val[0],2) + C2);
  cv::Mat covarM;
  cv::matchTemplate(cv::Mat_<float>(x), cv::Mat_<float>(y), covarM, CV_TM_CCOEFF);
  covarM/x.total();
  double covar;
  //x and y are same size, covarM is 1x1 matrix (only one matching)
  cv::minMaxLoc(covarM, nullptr, &covar, nullptr, nullptr, cv::Mat());
  s = (covar + (C2/2))/(stdDevX.val[0] + stdDevY.val[0] + (C2/2));
  return l*c*s;
}

//mean structural similarity index metric
cv::Scalar ImageQualityMetric::mssim( const cv::Mat& i1, const cv::Mat& i2)
{
  //SSIM is described more in-depth in the: “Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
  //“Image quality assessment: From error visibility to structural similarity,” IEEE Transactions on Image Processing,
  //vol. 13, no. 4, pp. 600-612, Apr. 2004.” article.
  //This will return a similarity index for each channel of the image.
  //This value is between zero and one, where one corresponds to perfect fit.
  //Unfortunately, the many Gaussian blurring is quite costly,
  //so while the PSNR may work in a real time like environment (24 frame per second)
  //this will take significantly more than to accomplish similar performance results.
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2   = I2.mul(I2);        // I2^2
    cv::Mat I1_2   = I1.mul(I1);        // I1^2
    cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2   =   mu1.mul(mu1);
    cv::Mat mu2_2   =   mu2.mul(mu2);
    cv::Mat mu1_mu2 =   mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    cv::Scalar mssim = cv::mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}
