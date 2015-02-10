/*
 * PDTools.cpp
 *
 *  Created on: Oct 25, 2013
 *      Author: dailos
 */
#include <iostream>
#include <limits>
#include "CustomException.h"
#include "PDTools.h"

cv::Mat conjComplex(const cv::Mat& A)
{
  try
  {  //try to implement with mixChannel function
    auto complexPairA = splitComplex(A);   //return matrix where every element is the conjugate
    return makeComplex(complexPairA.first, (complexPairA.second).mul(-1));
  }
  catch(...)
  {
    throw CustomException("conjComplex: Error");
  }
}

void partlyKnownDifferencesInPhaseConstraints(int M, int K, cv::Mat& Q2)
{
  //constraints equation: PartlyKnownDifferencesInPhase
  cv::Mat ce = cv::Mat::eye(M, M, cv::DataType<double>::type);
  for(int i = 1; i < K-1; ++i)
  {
    cv::vconcat(ce, cv::Mat::eye(M, M, cv::DataType<double>::type), ce);
  }
  
  cv::hconcat(ce, -1 * cv::Mat::eye((K-1)*M, (K-1)*M, cv::DataType<double>::type), ce);
  ce.row(0) = cv::abs(ce.row(0));
  ce.row(1) = cv::abs(ce.row(1));
  ce.row(2) = cv::abs(ce.row(2));
  
  cv::Mat Q, R;
  
  householder(ce.t(), Q, R);
  // extracts A columns, 1 (inclusive) to 3 (exclusive).
  //Mat B = A(Range::all(), Range(1, 3));
  //Select last Np colums of Q to become the constraints null space
  int Np = (M*K) - ce.rows;
  Q2 = Q(cv::Range::all(), cv::Range(Q.cols - Np, Q.cols ));
}

void householder(const cv::Mat &m, cv::Mat &Q, cv::Mat &R)
{
  if(m.cols>m.rows) throw CustomException("Assertion failed: cols<rows for QR decomposition");
  auto matrix_minor = [](cv::Mat x, int d) -> cv::Mat
  { //For d = 2, turns this into this:
    // 5, 5, 5, 5, 5      1, 0, 0, 0, 0
    // 5, 5, 5, 5, 5      0, 1, 0, 0, 0
    // 5, 5, 5, 5, 5  ->  0, 0, 5, 5, 5
    // 5, 5, 5, 5, 5      0, 0, 5, 5, 5
    // 5, 5, 5, 5, 5      0, 0, 5, 5, 5
  	cv::Mat m = cv::Mat::eye(x.size(), x.type());
  	for (int i = d; i < x.rows; i++)
    {
		  for (int j = d; j < x.cols; j++)
      {
	  		m.at<double>(i,j) = x.at<double>(i,j);
      }
    }
  
	  return m;
  };

  
	std::vector<cv::Mat> q;
	cv::Mat z = m.clone(), z1;
  
	for (int k = 0; k < m.cols && k < m.rows - 1; k++) 
  {
		cv::Mat e = cv::Mat::zeros(m.rows, 1, m.type()), x(m.rows, 1, m.type());
    double a;
		z1 = matrix_minor(z, k);
    
		z = z1.clone();
    z.col(k).copyTo(x);
    
		a = std::sqrt(x.dot(x));
    //If floating-point used, a should be opposite sign as the k-th coordinate m to avoid loss of significance
		if (m.at<double>(k,k) > 0) a = -a;  
		e.at<double>(k,0) = 1.0;
    e = x + (e*a);
		e = e / std::sqrt(e.dot(e));
    cv::Mat qi = cv::Mat::eye(e.rows, e.rows, e.type()) - 2.0 * e * e.t();
    
		q.push_back(qi.clone());
		z1 = qi * z;
		z1.copyTo(z);
	}
	
  //Q has dimensions of transpose of qi
  Q = cv::Mat::eye(q.front().cols, q.front().rows, q.front().type());
  //Multiply all elements of qi
  for(cv::Mat qi : q) Q = Q * qi.t(); 
  R = Q.t() * m;
}
  
/*
void shiftQuadrants(cv::Mat& I)
{
  // rearrange the quadrants of Fourier image  so that the origin is at the image center
  int cx = I.cols / 2;
  int cy = I.rows / 2;
  cv::Mat q0(I, cv::Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
  cv::Mat q1(I, cv::Rect(cx, 0, cx, cy)); // Top-Right
  cv::Mat q2(I, cv::Rect(0, cy, cx, cy)); // Bottom-Left
  cv::Mat q3(I, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

  cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);
}

  //GUI FEATURES
void showHistogram(const cv::Mat& src)
{
  if (src.channels() != 1)
  {
    throw CustomException("showHistogram: Only one channel image allowed.");
  }
  int histSize = 256;

  /// Set the ranges
  float range[] = { 0, 256 };
  const float* histRange = { range };

  bool uniform = true;
  bool accumulate = false;

  cv::Mat b_hist;

  /// Compute the histograms:
  calcHist(&src, 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, 	accumulate);

  // Draw the histograms
  int hist_w = 512;
  int hist_h = 400;
  int bin_w = cvRound((double) hist_w / histSize);

  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

  /// Normalize the result to [ 0, histImage.rows ]
  cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

  /// Draw for each channel
  for (int i = 1; i < histSize; i++)
  {
    cv::line( histImage, cv::Point(bin_w * (i - 1),
           hist_h - cvRound(b_hist.at<float> (i - 1))), cv::Point(bin_w * (i),
           hist_h - cvRound(b_hist.at<float> (i))), cv::Scalar(255, 0, 0), 2, 8, 0);
  }

  /// Display
  cv::namedWindow("Histogram", CV_WINDOW_AUTOSIZE);
  imshow("Histogram", histImage);
}

void showComplex(const cv::Mat& A, const std::string& txt, const bool& shiftQ, const bool& logScale)
{
  if (A.type() != CV_32FC2 && A.type() != CV_64FC2)
  {
    throw CustomException("complexImShow: Unsuported matrix type.");
  }

  cv::Mat planes[2];
  cv::Mat real, imag;
  split(A, planes);

  real = planes[0].clone();
  if (logScale)
  {
    log(real, real);
  }
  if (shiftQ)
  {
    shiftQuadrants(real);
  }
  normalize(real, real, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
  imshow("Real part: " + txt, real);

  imag = planes[1].clone();
  if (logScale)
  {
    log(imag, imag);
  }
  if (shiftQ)
  {
    shiftQuadrants(imag);
  }
  normalize(imag, imag, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
  imshow("Imaginary part: " + txt, imag);

  cv::Mat mag;
  magnitude(planes[0], planes[1], mag);
  mag += cv::Scalar::all(1); // switch to logarithmic scale
  if (logScale)
  {
    log(mag, mag);
  }
  mag = mag(cv::Rect(0, 0, mag.cols & -2, mag.rows & -2));
  if (shiftQ)
  {
    shiftQuadrants(mag);
  }

  normalize(mag, mag, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
  imshow("Magnitude: " + txt, mag);

  cv::waitKey();
}
*/

void shift(cv::Mat& I, cv::Mat& O, const int& cxIndex, const int& cyIndex)
{
  int ncols = I.cols;
  int nrows = I.rows;

  int cx = -cxIndex%ncols;
  if(cx < 0) cx = ncols+cx;

  int cy = -cyIndex%nrows;
  if(cy < 0) cy = nrows+cy;

  // rearrange the quadrants of image matrix
  cv::Mat q0, q1, q2, q3;

  if(cy > 0 && cx > 0)
  {
    q0 = I(cv::Rect(0, 0, cx, cy)).clone(); // Top-Left - Create a ROI per quadrant
  }
  if(cy > 0)
  {
    q1 = I(cv::Rect(cx, 0, ncols-cx, cy)).clone(); // Top-Right
  }
  if(cx > 0)
  {
    q2 = I(cv::Rect(0, cy, cx, nrows-cy)).clone(); // Bottom-Left
  }
  q3 = I(cv::Rect(cx, cy, ncols-cx, nrows-cy)).clone(); // Bottom-Right

  if(I.size() != O.size() || I.type() != O.type())
  {
    O = cv::Mat::zeros(I.size(), I.type());
  }
  //Copy to the place
  if(cy > 0 && cx > 0)
  {
    q0.copyTo(O(cv::Rect(ncols-cx,nrows-cy,cx,cy)));
  }
  if(cy > 0)
  {
    q1.copyTo(O(cv::Rect(0,nrows-cy,ncols-cx,cy))); // swap quadrant (Top-Right with Bottom-Left)
  }
  if(cx > 0)
  {
    q2.copyTo(O(cv::Rect(ncols-cx,0,cx,nrows-cy)));
  }
  q3.copyTo(O(cv::Rect(0,0,ncols-cx,nrows-cy)));
}

/**
 * @brief makeCanvas Makes composite image from the given images
 * @param vecMat Vector of Images.
 * @param windowHeight The height of the new composite image to be formed.
 * @param nRows Number of rows of images. (Number of columns will be calculated
 *              depending on the value of total number of images).
 * @return new composite image.
 */
/*
cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows)
{
  int N = vecMat.size();
  int edgeThickness = 10;
  int imagesPerRow = ceil(double(N) / nRows);
  int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
  int maxRowLength = 0;

  std::vector<int> resizeWidth;
  for (int i = 0; i < N;)
  {
    int thisRowLen = 0;
    for (int k = 0; k < imagesPerRow; k++)
    {
      double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
      int temp = int( ceil(resizeHeight * aspectRatio));
      resizeWidth.push_back(temp);
      thisRowLen += temp;
      if (++i == N) break;
    }
    if (thisRowLen > maxRowLength)
    {
      maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
    }
  }
  int windowWidth = maxRowLength;
  cv::Mat canvasImage(windowHeight, windowWidth, CV_64F, cv::Scalar(0.0));

  for (int k = 0, i = 0; i < nRows; i++)
  {
    int y = i * resizeHeight + (i + 1) * edgeThickness;
    int x_end = edgeThickness;
    for (int j = 0; j < imagesPerRow && k < N; k++, j++)
    {
      int x = x_end;
      cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
      cv::Mat target_ROI = canvasImage(roi);
      cv::resize(vecMat[k], target_ROI, target_ROI.size());
      x_end += resizeWidth[k] + edgeThickness;
    }
  }
  return canvasImage;
}

void writeOnImage(cv::Mat& img, const std::string& text)
{
  int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScale = 0.5;
  int thickness = 0.5;

  int baseline = 0;
  cv::Size textSize = cv::getTextSize(text, fontFace,
                              fontScale, thickness, &baseline);
  baseline += thickness;

  // center the text
  cv::Point textOrg((img.cols - textSize.width)/2,
                (img.rows + textSize.height)/2);

  // draw the box
//  cv::rectangle(img, textOrg + cv::Point(0, baseline),
//            textOrg + cv::Point(textSize.width, -textSize.height),
//            cv::Scalar(0,0,255));
  // ... and the baseline first
//  cv::line(img, textOrg + cv::Point(0, thickness),
//       textOrg + cv::Point(textSize.width, thickness),
//       cv::Scalar(0, 0, 255));

  // then put the text itself
  cv::putText(img, text, textOrg, fontFace, fontScale,
          cv::Scalar::all(255), thickness, 8);

}
*/
unsigned int optimumSideLength(const unsigned int& minimumLength, const double& radiousLength)
{ //Enlarge the image size if the a circle with radious length doesn't fit in it
  double diff = (2*radiousLength) - (minimumLength-2);
  unsigned int optimumLength = minimumLength;
  if(diff >= 0)
  {
    optimumLength = minimumLength + std::ceil(diff/2) * 2;
  }
  return optimumLength;
}

cv::Mat centralROI(const cv::Mat& im, const cv::Size& roiSize, cv::Mat& roi)
{
  cv::Point roiPosition((im.cols/2)-(roiSize.width/2), (im.rows/2)-(roiSize.height/2));
  //roi = im(cv::Rect(roiPosition, roiSize));
  return im(cv::Rect(roiPosition, roiSize));
}

cv::Mat selectCentralROI(const cv::Mat& im, const cv::Size& roiSize)
{
  cv::Point roiPosition((im.cols/2)-(roiSize.width/2), (im.rows/2)-(roiSize.height/2));
  return im(cv::Rect(roiPosition, roiSize));
}

cv::Mat takeoutImageCore(const cv::Mat& im, const unsigned int& imageCoreSize)
{  //Extract the center part of the image of size imageCoreSize
  if(im.cols == im.rows)
  {
    cv::Point coreCornerPosition((im.cols/2)-(imageCoreSize/2), (im.cols/2)-(imageCoreSize/2));
    return im(cv::Rect(coreCornerPosition, cv::Size(imageCoreSize,imageCoreSize))).clone();
  }
  else
  {
    throw CustomException("takeoutImageCore: Must have same number of rows and cols to extract central core.");
  }
}


cv::Mat crosscorrelation_direct(const cv::Mat& A, const cv::Mat& B)
{  //For testing pourposes only
  cv::Mat aPadded, bPadded;
  cv::copyMakeBorder(A, aPadded, 0, A.rows, 0, A.cols, cv::BORDER_CONSTANT, cv::Scalar(0.0, 0.0));
  cv::copyMakeBorder(B, bPadded, 0, B.rows, 0, B.cols, cv::BORDER_CONSTANT, cv::Scalar(0.0, 0.0));

  cv::Mat C = cv::Mat::zeros(aPadded.size(), aPadded.type());
  cv::Mat conjA = conjComplex(aPadded);
  for(int i=0; i < aPadded.cols; ++i)              // rows
  {
    for(int j=0; j < aPadded.rows; ++j)          // columns
    {
      cv::Mat shifted, prod;
      shift(conjA,shifted,aPadded.cols-i,aPadded.rows-j);
      cv::mulSpectrums(shifted,bPadded,prod, cv::DFT_COMPLEX_OUTPUT);
      cv::Scalar s = cv::sum(prod);
      C.at<std::complex<float> >(i,j) = std::complex<float>(s.val[0],s.val[1]);
    }
  }
  return C;
}


void conv_flaw(const cv::Mat& imgOriginal, const cv::Mat& kernel, cv::Mat& out, const bool& corr)
{
  cv::Mat source;
  imgOriginal.copyTo(source);
  cv::Mat kernelPadded = cv::Mat::zeros(source.size(), source.type());
  if(kernel.size().height > kernelPadded.size().height || kernel.size().width > kernelPadded.size().width)
  {
    throw CustomException("Kernel padded bigger than image.");
  }
  kernel.copyTo(selectCentralROI(kernelPadded, kernel.size()));
  //Divided by 2.0 instead of 2 to consider the result as double instead of as an int
  //The +1 in the shift changes slightly the finest plane in the wavelet,
  shift(kernelPadded, kernelPadded, std::ceil(kernelPadded.cols/2.0), std::ceil(kernelPadded.rows/2.0));

  cv::Mat kernelPadded_ft, input_ft, output_ft;
  cv::dft(kernelPadded, kernelPadded_ft, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  cv::dft(source, input_ft, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  cv::mulSpectrums(input_ft, kernelPadded_ft.mul(kernelPadded.total()), output_ft, cv::DFT_COMPLEX_OUTPUT, corr);
  cv::idft(output_ft, out, cv::DFT_REAL_OUTPUT);
}

void convolveDFT(const cv::Mat& imgOriginal, const cv::Mat& kernel, cv::Mat& out, const bool& corr, const bool& full)
{ //this method is also valid for complex image and kernel, set DFT_COMPLEX_OUTPUT then
  //convolution in fourier space, keep code for future use
  //CONVOLUTION_FULL: Return the full convolution, including border
  //to completeley emulate filter2D operation, image should be first double sized and then cut back to origianl size
  cv::Mat source, kernelPadded;
  const int marginSrcTop = corr ? std::ceil((kernel.rows-1)/2.0) : std::floor((kernel.rows-1)/2.0);
  const int marginSrcBottom = corr ? std::floor((kernel.rows-1)/2.0) : std::ceil((kernel.rows-1)/2.0);
  const int marginSrcLeft = corr ? std::ceil((kernel.cols-1)/2.0) : std::floor((kernel.cols-1)/2.0);
  const int marginSrcRight = corr ? std::floor((kernel.cols-1)/2.0) : std::ceil((kernel.cols-1)/2.0);
  cv::copyMakeBorder(imgOriginal, source, marginSrcTop, marginSrcBottom, marginSrcLeft, marginSrcRight, cv::BORDER_CONSTANT);

  const int marginKernelTop = std::ceil((source.rows-kernel.rows)/2.0);
  const int marginKernelBottom = std::floor((source.rows-kernel.rows)/2.0);
  const int marginKernelLeft = std::ceil((source.cols-kernel.cols)/2.0);
  const int marginKernelRight = std::floor((source.cols-kernel.cols)/2.0);
  cv::copyMakeBorder(kernel, kernelPadded, marginKernelTop, marginKernelBottom, marginKernelLeft, marginKernelRight, cv::BORDER_CONSTANT);

  //Divided by 2.0 instead of 2 to consider the result as double instead of as an int
  //The +1 in the shift changes slightly the finest plane in the wavelet,
  shift(kernelPadded, kernelPadded, std::ceil(kernelPadded.cols/2.0), std::ceil(kernelPadded.rows/2.0));

  cv::Mat kernelPadded_ft, input_ft, output_ft;
  cv::dft(kernelPadded, kernelPadded_ft, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  cv::dft(source, input_ft, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  cv::mulSpectrums(input_ft, kernelPadded_ft.mul(kernelPadded.total()), output_ft, cv::DFT_COMPLEX_OUTPUT, corr);
  cv::idft(output_ft, out, cv::DFT_REAL_OUTPUT);
  if(!full)
  {
    //colRange and rowRange are semi-open intervals. first included, last is not
    //this frist option is what i think should be the correct one, but the next is what filter2D function gives for this inputs
//    out = out.colRange(marginSrcLeft, out.cols - marginSrcRight).
//              rowRange(marginSrcTop, out.rows - marginSrcBottom);
        out = out.colRange(marginSrcRight, out.cols - marginSrcLeft).
                  rowRange(marginSrcBottom, out.rows - marginSrcTop);
  }
}

void convolve(const cv::Mat& imgOriginal, const cv::Mat& kernel, cv::Mat& out, const bool& corr, const bool& full)
{
  //kernel size is supposed to be smaller than original image
  //output image size is same as input image. EFFICIENT VERSION!!
  //Place anchor at the center by default
  if(imgOriginal.cols < kernel.cols || imgOriginal.rows < kernel.rows)
  {
    throw CustomException("Convolution kernel should always be smaller than image.");
  }
  cv::Point anchor = cv::Point(kernel.cols - std::ceil(kernel.cols/2.0), kernel.rows - std::ceil(kernel.rows/2.0));  //corr = true
  if(full)
  {
    cv::Mat source;
    const int marginSrcTop = corr ? std::floor((kernel.rows-1)/2.0) : std::ceil((kernel.rows-1)/2.0);
    const int marginSrcBottom = corr ? std::ceil((kernel.rows-1)/2.0) : std::floor((kernel.rows-1)/2.0);
    const int marginSrcLeft = corr ? std::floor((kernel.cols-1)/2.0) : std::ceil((kernel.cols-1)/2.0);
    const int marginSrcRight = corr ? std::ceil((kernel.cols-1)/2.0) : std::floor((kernel.cols-1)/2.0);
    cv::copyMakeBorder(imgOriginal, source, marginSrcTop, marginSrcBottom, marginSrcLeft, marginSrcRight, cv::BORDER_CONSTANT);
    if(corr)
    {
      cv::filter2D(source, out, source.depth(), kernel, anchor, 0, cv::BORDER_CONSTANT);
    }
    else
    {//filter2D applies correlation by default, so kernerl and anchor must be changed accordingly
      cv::Mat mod_kernel;
      cv::flip(kernel, mod_kernel, -1);
      anchor = cv::Point(kernel.cols - anchor.x - 1, kernel.rows - anchor.y - 1);  //corr = false
      cv::filter2D(source, out, source.depth(), mod_kernel, anchor, 0, cv::BORDER_CONSTANT);
    }
  }
  else
  {
    if(corr)
    {
      cv::filter2D(imgOriginal, out, imgOriginal.depth(), kernel, anchor, 0, cv::BORDER_CONSTANT);
    }
    else
    {
      cv::Mat mod_kernel;
      cv::flip(kernel, mod_kernel, -1);
      anchor = cv::Point(kernel.cols - anchor.x - 1, kernel.rows - anchor.y - 1);  //corr = false
      cv::filter2D(imgOriginal, out, imgOriginal.depth(), mod_kernel, anchor, 0, cv::BORDER_CONSTANT);
    }
  }
}

cv::Mat crosscorrelation(const cv::Mat& A, const cv::Mat& B)
{
  if (A.channels() != 2 || B.channels() != 2)
  {
    throw CustomException("crosscorrelation: Must be two two-channel images.");
  }

  cv::Mat aPadded;
  cv::Mat bPadded;
  cv::Mat doubleA, doubleB;
/*
  //The following expands the image to an optimal size in order to be the fourier transform efficient
   * It is not used right now, since the image is double-sized to apply crosscorrelation properly
  int m = getOptimalDFTSize( A.rows );
  int n = getOptimalDFTSize( A.cols ); // on the border add zero values
  copyMakeBorder(A, aPadded, 0, m - A.rows, 0, n - A.cols, BORDER_CONSTANT, Scalar(0,0));
  copyMakeBorder(B, bPadded, 0, p - B.rows, 0, q - B.cols, BORDER_CONSTANT, Scalar(0,0));
*/
  //REMEMBER!!  The result of the correlation is twice the size of the input arrays!
  //There must be enough space for the overlapping of both functions
  cv::copyMakeBorder(A, aPadded, 0, A.rows, 0, A.cols, cv::BORDER_CONSTANT, cv::Scalar(0.0, 0.0));
  cv::copyMakeBorder(B, bPadded, 0, B.rows, 0, B.cols, cv::BORDER_CONSTANT, cv::Scalar(0.0, 0.0));

  //CAUTION!! Know differences between: DFT_COMPLEX_OUTPUT, DFT_SCALE, DFT_REAL_OUTPUT
  cv::dft(aPadded, aPadded, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);
  cv::dft(bPadded, bPadded, cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE);

  cv::Mat C, tmpC;
  //CAUTION!! Know differences between: DFT_COMPLEX_OUTPUT, DFT_SCALE, DFT_REAL_OUTPUT
  bool conjugateB(true);  //optional parameter, false by default :: set this parameter to false to turn this operation into convolution
  cv::mulSpectrums(aPadded, bPadded.mul(aPadded.rows * aPadded.cols), tmpC, cv::DFT_COMPLEX_OUTPUT, conjugateB);

  //Note None of dft and idft scales the result by default.
  //So, you should pass DFT_SCALE to one of dft or idft explicitly to make these transforms mutually inverse.
  cv::idft(tmpC, C, cv::DFT_COMPLEX_OUTPUT);

  //REMEMBER NORMALIZE!! When calculating OTF, the value at origin must be equal to unity!! energy conservation
  return C;
}

cv::Mat divComplex(const cv::Mat& A, const cv::Mat& B)
{
  if (A.channels() != 2 || B.channels() != 2 )
  {
    throw CustomException("divComplex: Must be two-channel image.");
  }
  auto sA = splitComplex(A);
  auto sB = splitComplex(B);

  cv::Mat den = (sB.first).mul(sB.first) + (sB.second).mul(sB.second);

  return makeComplex( ( (sA.first).mul(sB.first)  + (sA.second).mul(sB.second) )/den,
                       ( (sA.second).mul(sB.first) - (sA.first).mul(sB.second)  )/den  );
      //In order to keep generic image type, it's a matrix instead of a complex value
}

cv::Mat normComplex(const cv::Mat& A, cv::Mat& out)
{//try to implement divComplex
  try
  {
    if (A.channels() != 2)
    {
        throw CustomException("normComplex: Must be two-channel image.");
    }
    //a+bi/c+di=(ac+bd/c^2+d^2)+(bc-ad/c^2+d^2)i
    //Divide every matrix element by the complex value at the origin (frequency 0,0)
    auto sA = splitComplex(A);
    cv::Mat norm = cv::repeat(A.col(0).row(0), A.rows, A.cols);
    
    
    auto sF = splitComplex(norm);
    //Zero value the imaginary part of the normalizations factor, it should be zero anyway
    sF.second = cv::Mat::zeros(sF.first.size(), sF.first.type());
    
    cv::Mat den = (sF.first).mul(sF.first) + (sF.second).mul(sF.second);
    out = makeComplex( ( (sA.first).mul(sF.first)  + (sA.second).mul(sF.second) )/den,
                     ( (sA.second).mul(sF.first) - (sA.first).mul(sF.second)  )/den  );
    //In order to keep generic image type, it's a matrix instead of a complex value
    return makeComplex(sF.first, sF.second);
  }
  catch(...)
  {
    throw;
  }
}

cv::Mat absComplex(const cv::Mat& complexI)
{
  if(complexI.channels() != 2)
  {
    throw CustomException("absComplex: Input must be a two channel image.");
  }

  cv::Mat planes[2];
  cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
  cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
  return planes[0];
}

long factorial(const long& theNumber)
{
  try
  {
    long i;
    long f = 1;

    for (i = 1; i<=theNumber; ++i)
    {
      f = f*i;
    }
    return f;
  }
  catch (std::exception& ex)
  {
    throw;
  }
}

std::pair<cv::Mat, cv::Mat> splitComplex(const cv::Mat& I)
{  //takes real part of complex matrix
  if(I.channels() != 2)
  {
    throw CustomException("splitComplex: Input must be a two channel matrix.");
  }
  cv::Mat planes[2];
  cv::split(I, planes);
  return std::make_pair(planes[0],planes[1]);
}

cv::Mat makeComplex(const cv::Mat& real, const cv::Mat& imag)
{//try to implement with cv::merge function
  // implement using imag = cv::noArray()
  cv::Mat C;
  if(real.channels() == 1 && imag.channels() == 1)
  {
    cv::Mat planes[] = {real, imag};
    cv::merge(planes, 2, C);
    return C;
  }
  else
  {
    throw CustomException("makeComplex: It should be both real and imag single channel images.");
  }
}

cv::Mat makeComplex(const cv::Mat& real)
{
  cv::Mat C;
  if(real.channels() == 1)
  {
    cv::Mat planes[] = {real, cv::Mat::zeros(real.size(), real.type())};
    cv::merge(planes, 2, C);
  }
  else if(real.channels() == 2)
  {  //real contains the imaginary part inside
    C = real;
  }
  else
  {
    throw CustomException("makeComplex: Argument can be both single or two channel image.");
  }

  return C;
}

//from phasecorr.cpp within opencv library code
void fftShift(cv::Mat& out)
{
    if(out.rows == 1 && out.cols == 1)
    {
        // trivially shifted.
        return;
    }

    std::vector<cv::Mat> planes;
    cv::split(out, planes);

    int xMid = out.cols >> 1;
    int yMid = out.rows >> 1;

    bool is_1d = xMid == 0 || yMid == 0;

    if(is_1d)
    {
        xMid = xMid + yMid;

        for(size_t i = 0; i < planes.size(); i++)
        {
            cv::Mat tmp;
            cv::Mat half0(planes[i], cv::Rect(0, 0, xMid, 1));
            cv::Mat half1(planes[i], cv::Rect(xMid, 0, xMid, 1));

            half0.copyTo(tmp);
            half1.copyTo(half0);
            tmp.copyTo(half1);
        }
    }
    else
    {
        for(size_t i = 0; i < planes.size(); i++)
        {
            // perform quadrant swaps...
            cv::Mat tmp;
            cv::Mat q0(planes[i], cv::Rect(0,    0,    xMid, yMid));
            cv::Mat q1(planes[i], cv::Rect(xMid, 0,    xMid, yMid));
            cv::Mat q2(planes[i], cv::Rect(0,    yMid, xMid, yMid));
            cv::Mat q3(planes[i], cv::Rect(xMid, yMid, xMid, yMid));

            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);

            q1.copyTo(tmp);
            q2.copyTo(q1);
            tmp.copyTo(q2);
        }
    }

    cv::merge(planes, out);
}


//from phasecorr.cpp within opencv library code
void divSpectrums(const cv::Mat& srcA, const cv::Mat& srcB, cv::Mat& dst, int flags, bool conjB)
{
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;

    if( !(type == srcB.type() && srcA.size() == srcB.size()) ) throw CustomException("divSpectrums: Wrong dimensions.");
  
    if( !(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2) ) throw CustomException("divSpectrums: Wrong input type.");

    dst.create( srcA.rows, srcA.cols, type );
    

    bool is_1d = (flags & cv::DFT_ROWS) || (rows == 1 || (cols == 1 &&
             srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

    if( is_1d && !(flags & cv::DFT_ROWS) )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if( depth == CV_32F )
    {
        const float* dataA = srcA.ptr<float>();
        const float* dataB = srcB.ptr<float>();
        float* dataC = dst.ptr<float>();
        float eps = FLT_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                                       (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA]*dataB[j*stepB] +
                                    (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] -
                                    (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j+1)*stepC] = (float)(im / denom);
                    }
                else
                    for( j = 1; j <= rows - 2; j += 2 )
                    {

                        double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                                       (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA]*dataB[j*stepB] -
                                    (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] +
                                    (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j+1)*stepC] = (float)(im / denom);
                    }
                if( k == 1 )
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
        {
            if( is_1d && cn == 1 )
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( cols % 2 == 0 )
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if( !conjB )
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = (double)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
                    double re = (double)(dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1]);
                    double im = (double)(dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j+1] = (float)(im / denom);
                }
            else
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = (double)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
                    double re = (double)(dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1]);
                    double im = (double)(dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j+1] = (float)(im / denom);
                }
        }
    }
    else
    {
        const double* dataA = srcA.ptr<double>();
        const double* dataB = srcB.ptr<double>();
        double* dataC = dst.ptr<double>();
        double eps = DBL_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = dataB[j*stepB]*dataB[j*stepB] +
                                       dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                        double re = dataA[j*stepA]*dataB[j*stepB] +
                                    dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = dataA[(j+1)*stepA]*dataB[j*stepB] -
                                    dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j+1)*stepC] = im / denom;
                    }
                else
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = dataB[j*stepB]*dataB[j*stepB] +
                                       dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                        double re = dataA[j*stepA]*dataB[j*stepB] -
                                    dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = dataA[(j+1)*stepA]*dataB[j*stepB] +
                                    dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j+1)*stepC] = im / denom;
                    }
                if( k == 1 )
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
        {
            if( is_1d && cn == 1 )
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( cols % 2 == 0 )
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if( !conjB )
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                    double re = dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1];
                    double im = dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1];
                    dataC[j] = re / denom;
                    dataC[j+1] = im / denom;
                }
            else
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                    double re = dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1];
                    double im = dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1];
                    dataC[j] = re / denom;
                    dataC[j+1] = im / denom;
                }
        }
    }
}

void magSpectrums( const cv::Mat& src, cv::Mat& dst)
{
    //Mat src = _src.getMat();
    int depth = src.depth(), cn = src.channels(), type = src.type();
    int rows = src.rows, cols = src.cols;
    int j, k;

    if( !(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2) ) throw CustomException("magSpectrums: Wrong input type.");

    if(src.depth() == CV_32F)
        dst.create( src.rows, src.cols, CV_32FC1 );
    else
        dst.create( src.rows, src.cols, CV_64FC1 );

    //Mat dst = _dst.getMat();
    dst.setTo(0);//Mat elements are not equal to zero by default!

    bool is_1d = (rows == 1 || (cols == 1 && src.isContinuous() && dst.isContinuous()));

    if( is_1d )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if( depth == CV_32F )
    {
        const float* dataSrc = src.ptr<float>();
        float* dataDst = dst.ptr<float>();

        size_t stepSrc = src.step/sizeof(dataSrc[0]);
        size_t stepDst = dst.step/sizeof(dataDst[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( rows % 2 == 0 )
                    dataDst[(rows-1)*stepDst] = dataSrc[(rows-1)*stepSrc]*dataSrc[(rows-1)*stepSrc];

                for( j = 1; j <= rows - 2; j += 2 )
                {
                    dataDst[j*stepDst] = (float)std::sqrt((double)dataSrc[j*stepSrc]*dataSrc[j*stepSrc] +
                                                          (double)dataSrc[(j+1)*stepSrc]*dataSrc[(j+1)*stepSrc]);
                }

                if( k == 1 )
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for( ; rows--; dataSrc += stepSrc, dataDst += stepDst )
        {
            if( is_1d && cn == 1 )
            {
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( cols % 2 == 0 )
                    dataDst[j1] = dataSrc[j1]*dataSrc[j1];
            }

            for( j = j0; j < j1; j += 2 )
            {
                dataDst[j] = (float)std::sqrt((double)dataSrc[j]*dataSrc[j] + (double)dataSrc[j+1]*dataSrc[j+1]);
            }
        }
    }
    else
    {
        const double* dataSrc = src.ptr<double>();
        double* dataDst = dst.ptr<double>();

        size_t stepSrc = src.step/sizeof(dataSrc[0]);
        size_t stepDst = dst.step/sizeof(dataDst[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( rows % 2 == 0 )
                    dataDst[(rows-1)*stepDst] = dataSrc[(rows-1)*stepSrc]*dataSrc[(rows-1)*stepSrc];

                for( j = 1; j <= rows - 2; j += 2 )
                {
                    dataDst[j*stepDst] = std::sqrt(dataSrc[j*stepSrc]*dataSrc[j*stepSrc] +
                                                   dataSrc[(j+1)*stepSrc]*dataSrc[(j+1)*stepSrc]);
                }

                if( k == 1 )
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for( ; rows--; dataSrc += stepSrc, dataDst += stepDst )
        {
            if( is_1d && cn == 1 )
            {
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( cols % 2 == 0 )
                    dataDst[j1] = dataSrc[j1]*dataSrc[j1];
            }

            for( j = j0; j < j1; j += 2 )
            {
                dataDst[j] = std::sqrt(dataSrc[j]*dataSrc[j] + dataSrc[j+1]*dataSrc[j+1]);
            }
        }
    }
}


