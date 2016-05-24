/*
 * ToolBox.h
 *
 *  Created on: Oct 25, 2013
 *      Author: dailos
 */

#ifndef TOOLBOX_H_
#define TOOLBOX_H_
#include <iostream>
#include "opencv2/opencv.hpp"

void divideIntoTiles(const cv::Size& dim, const unsigned int& pixelsBetweenTiles, const unsigned int& tileSize, std::vector<std::pair<cv::Range,cv::Range> >& tileRngs);
void shuffleRows(const cv::Mat &matrix, cv::Mat& shuffleMatrix);
double rms(const cv::Mat& A, const cv::Mat& mask = cv::Mat());
void reorderColumns(const cv::Mat& A, const unsigned int& slice, cv::Mat& Ar);
cv::Mat conjComplex(const cv::Mat& A);
void partlyKnownDifferencesInPhaseConstraints(int M, int K, cv::Mat& Q2);
void householder(const cv::Mat &m, cv::Mat &Q, cv::Mat &R);
void shift(cv::Mat& I, cv::Mat& O, const int& cx, const int& cy);
cv::Mat crosscorrelation(const cv::Mat& A, const cv::Mat& B);
cv::Mat crosscorrelation_direct(const cv::Mat& A, const cv::Mat& B);
void convolve(const cv::Mat& imgOriginal, const cv::Mat& kernel, cv::Mat& out, const bool& corr = false, const bool& full = false);
void conv_flaw(const cv::Mat& imgOriginal, const cv::Mat& kernel, cv::Mat& out, const bool& corr = false);
void convolveDFT(const cv::Mat& imgOriginal, const cv::Mat& kernel, cv::Mat& out, const bool& corr = false, const bool& full = false);
unsigned int optimumSideLength(const unsigned int& minimumLength, const double& radiousLength);
cv::Mat takeoutImageCore(const cv::Mat& im, const unsigned int& imageCoreSize);
cv::Mat selectCentralROI(const cv::Mat& im, const cv::Size& roiSize);
cv::Mat centralROI(const cv::Mat& im, const cv::Size& roiSize, cv::Mat& roi);
//void writeOnImage(cv::Mat& img, const std::string& text);
//cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows);
cv::Mat absComplex(const cv::Mat& complexI);
cv::Mat normComplex(const cv::Mat& A, cv::Mat& out);
cv::Mat divComplex(const cv::Mat& A, const cv::Mat& B);
//void shiftQuadrants(cv::Mat& I);
//void showComplex(const cv::Mat& A, const std::string& txt, const bool& shiftQ = true, const bool& logScale = false);
//void showHistogram(const cv::Mat& src);
std::pair<cv::Mat, cv::Mat> splitComplex(const cv::Mat& I);
cv::Mat makeComplex(const cv::Mat& real, const cv::Mat& imag);
cv::Mat makeComplex(const cv::Mat& real);
long factorial(const long& theNumber);


//methods from phasecorr.cpp
void fftShift(cv::Mat& out);
void divSpectrums(const cv::Mat& srcA, const cv::Mat& srcB, cv::Mat& dst, int flags, bool conjB = false);
void magSpectrums(const cv::Mat& src, cv::Mat& dst);

#endif /* TOOLBOX_H_ */
