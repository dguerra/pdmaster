/*
 * TestRoom.h
 *
 *  Created on: Feb 18, 2014
 *      Author: dailos
 */

#ifndef TESTROOM_H_
#define TESTROOM_H_
#include "opencv2/opencv.hpp"


template<class T>
cv::Mat createRandomMatrix(const unsigned int& xSize, const unsigned int& ySize);
void test_jacobian();
void leastSquearesExperiment();
bool test_BSL();
void test_SparseRecovery();
bool test_LinearizedBregmanAlgorithmDenoising();
void test_noiseFilter();
void test_generizedPupilFunctionVsOTF();
void test_udwd_spectrums();
void test_zernike_wavelets_decomposition();
void test_wavelet_zernikes_decomposition();
void test_getNM();
bool test_convolveDFT_vs_crosscorrelation();
bool test_shift();
void test_conv_flaw();
void test_wavelets();
bool test_conjComplex();
bool test_minimization();
bool test_minQ2();
void test_covarianceMatrix();
bool test_nonsmoothConvexOptimization();
bool test_crosscorrelation();
bool test_specular();
void test_selectCentralROI();
void test_AWMLE();
bool test_fourier();
bool test_normalization();
bool test_zernikes();
void test_SVD();
void test_flip();
void test_QualityMetric();
void test_ErrorMetric();
void test_Optics();
bool test_NoiseEstimator();
void test_Noise();
void test_phaseMapZernikeSum();
void test_divComplex();
void test_convolve();
void test_convolution_algo();
void test_filter2D();
#endif /* TESTROOM_H_ */
