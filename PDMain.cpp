//============================================================================
// Name        : PDMain.cpp
// Author      : Dailos G
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "TestRoom.h"
#include "SubimageLayout.h"
#include "FITS.h"
//#include "ConvexOptimization.h"
//#include "Metric.h"
#include "ToolBox.h"

#include "Zernike.h"
#include <chrono>
#include <cmath>
#include "SparseRecovery.h"
#include "PhaseScreen.h"

int main()
{
  try
  {
    std::cout << "Hello" << std::endl;
    SubimageLayout subimageLayout;
    //test_minQ2();
    //test_covarianceMatrix();
    subimageLayout.computerGeneratedImage();
  }
  catch (cv::Exception const & e)
  {
    std::cerr << "OpenCV exception: " << e.what() << std::endl;
  }

  return 0;
}

