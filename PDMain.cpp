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
//#include "Metric.h"Información
#include "ToolBox.h"

#include "Zernike.h"
#include <chrono>
#include <cmath>
#include "SparseRecovery.h"
#include "PhaseScreen.h"
#include "Regression.h"
#include "DataSet.h"
#include <fstream>
int main()
{
  try
  {
    std::cout << "Hello" << std::endl;
    SubimageLayout subimageLayout;

    subimageLayout.computerGeneratedImage();    
    
    //subimageLayout.dataSimulator();

  }
  catch (cv::Exception const & e)
  {
    std::cerr << "OpenCV exception: " << e.what() << std::endl;
  }

  return 0;
}


