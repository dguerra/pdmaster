//============================================================================
// Name        : PDPlayground.cpp
// Author      : Dailos
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <functional>   // std::minus
#include <numeric>      // std::accumulate
#include <math.h>
#include <tuple>
#include <string>
#include <vector>
#include <limits>
#include <complex>
#include <algorithm>
#include "Zernikes.h"
#include "NoiseEstimator.h"
#include "WaveletTransform.h"
#include "PDTools.h"
#include "Optimization.h"
#include "OpticalSystem.h"
#include "ErrorMetric.h"
#include "opencv2/opencv.hpp"
#include "TestRoom.h"
#include "TelescopeSettings.h"
#include "WavefrontSensor.h"
#include "FITS.h"
#include "SubimageLayout.h"
#include "AWMLE.h"
#include "Benchmark.h"

#include <chrono>

//to compare the results mean squared error (MSE) and the structural similarity index (SSIM)

int main()
{
  try
  {
    SubimageLayout subimageLayout;
    subimageLayout.navigateThrough();

//    test_AWMLE();
  }
  catch (cv::Exception const & e)
  {
    std::cerr << "OpenCV exception: " << e.what() << std::endl;
  }

  return 0;
}
