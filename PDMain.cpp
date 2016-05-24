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
//#include "ConvexOptimization.h"
//#include "Metric.h"
#include "ToolBox.h"

#include "BasisRepresentation.h"
#include <chrono>
#include <cmath>
#include "SparseRecovery.h"

int main()
{
  try
  {
    //test_BSL();
    SubimageLayout subimageLayout;
    subimageLayout.computerGeneratedImage();
    
  }
  catch (cv::Exception const & e)
  {
    std::cerr << "OpenCV exception: " << e.what() << std::endl;
  }

  return 0;
}
