//============================================================================
// Name        : PDMain.cpp
// Author      : Dailos
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "TestRoom.h"
#include "SubimageLayout.h"
#include "FITS.h"
#include "Minimization.h"
#include "Metric.h"
#include "PDTools.h"
#include <chrono>

//to compare the results mean squared error (MSE) and the structural similarity index (SSIM)

int main()
{
  try
  {
    SubimageLayout subimageLayout;
    subimageLayout.navigateThrough();
  }
  catch (cv::Exception const & e)
  {
    std::cerr << "OpenCV exception: " << e.what() << std::endl;
  }

  return 0;
}
