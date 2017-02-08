/*
 * Regression.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: dailos
 */

#include "Regression.h"

Regression::Regression()
{
  // TODO default constructor
}


Regression::~Regression()
{
  // TODO Auto-generated destructor stub
}

void Regression::test_1D()
{
  //We have N=15 samples from y=sin(x) in the interval [0,2π] and add gaussian noise
  unsigned int N = 5;  //Number of input samples
  unsigned int M = 5;  //Number of output samples
  const double pi = 3.14159265358979323846;  /* pi */
  double startPoint(0.0);
  double endPoint(2.0 * pi);
  
  
  double stepSizeN((endPoint-startPoint)/double(N-1)); //Distance between inputs equal distance
  
  cv::Mat x,t;
  cv::theRNG() = cv::RNG( cv::getTickCount() );
  cv::RNG& rng = cv::theRNG();
  
  //create input samples plus noise
  for(double xi=startPoint; xi<=endPoint; xi=xi+stepSizeN)
  {
    x.push_back(xi);
    t.push_back(sin(xi)+rng.gaussian(0.2));
  }
  std::cout << "x: " << x.t() << std::endl;
  std::cout << "t: " << t.t() << std::endl;
  
  //For the output we decide to have M wights for every RBF non-linear function at equal distance 
  //Suppose now I want to modeled the system with a combination of M potentially nonlinear basis functions such as radial basis functions, RBF
  auto rbf = [](const double& x, const double& xm, const double& r) -> double
  { //exp(-(x-xm)^2/r^2)
    return std::exp(-std::pow(x-xm,2.0)/(r*r));
  }; 

  //
  double stepSizeM((endPoint-startPoint)/double(M-1)); //Distance between base functions equal distance
  
  
  std::vector<cv::Mat> phi_v;
  for(double xm=startPoint; xm<=endPoint; xm=xm+stepSizeM)
  {
    cv::Mat phi_m;
    //We use the same locations for the center of the radial functions
    for(double xi=startPoint; xi<=endPoint; xi=xi+stepSizeN)
    {
      phi_m.push_back(rbf(xi,xm,1.0));
    }
    phi_v.push_back(phi_m);
  }
  cv::Mat phi;
  cv::hconcat(phi_v, phi);


  //Apply least squares to find the weights
  cv::Mat w = (phi.t()*phi+0.5*cv::Mat::eye(N,N,cv::DataType<double>::type)).inv()*phi.t()*t;   //Ridge regression
  //cv::Mat w = (phi.t()*phi).inv()*phi.t()*t;   //Least squares

  int input_neurons = 1;     // input layer size
  int hidden_neurons = 5;  // hidden layer size
  int output_neurons = 1;  // output layer size
  
  cv::Ptr<cv::ml::ANN_MLP> neural_network = cv::ml::ANN_MLP::create();
  neural_network->setTrainMethod(cv::ml::ANN_MLP::BACKPROP);
  neural_network->setBackpropMomentumScale(0.1);
  neural_network->setBackpropWeightScale(0.05);
  neural_network->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, (int)10000, 1e-6));
  
  std::vector<int> layerSizes = { input_neurons, hidden_neurons, output_neurons };
  
  //# samples = Size[ inputLayerSize, numSamples ] -> CV_32F
  cv::Mat samples;
  x.convertTo(samples, CV_32F);
  //# responses = Size[ outputLayerSize, numSamples ] -> CV_32F
  cv::Mat responses;
  t.convertTo(responses, CV_32F);
  
  neural_network->setLayerSizes(layerSizes);
  neural_network->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1); 
  
  cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE, responses);
  
  neural_network->train( tData );

//  cv::Ptr<cv::TrainData> train_data = TrainData::loadFromCSV("data.csv", 10, 7, 12);

  std::cout << "END" << std::endl;  
  
/*
  //Show result: (prediction)
  cv::Mat result;
  cv::Mat cur;
  for(double cursor=startPoint;cursor<=endPoint;cursor=cursor+0.01)
  {
    double val(0.0);
    int i(0);
    for(double xm=startPoint;  xm<=endPoint; xm=xm+stepSize)
    {
      val = val + w.at<double>(i++,0) * rbf(cursor,xm,1.0);
    }
    result.push_back(val);
    cur.push_back(cursor);
  }

  //std::cout << "phi: " << phi << std::endl;
  cv::Mat plot_data, plot_result;
  Plot2d plot1( x,t );
  Plot2d plot2( cur,result );
  //cv::plot->setPlotBackgroundColor( cv::Scalar( 50, 50, 50 ) ); // i think it is not implemented yet
  plot1.setPlotLineColor( cv::Scalar( 255, 255, 255 ) );
  plot1.setPlotLineWidth(2);
  plot1.setNeedPlotLine(false);
  plot1.render( plot_data );

  plot2.setPlotLineColor( cv::Scalar( 255, 255, 255 ) );
  plot2.setPlotLineWidth(2);
  plot2.setNeedPlotLine(false);
  plot2.render( plot_result );
  
  cv::Mat gray_result;
  cv::cvtColor(plot_result, gray_result, CV_BGR2GRAY);
  writeFITS(gray_result, "../gray_result.fits");
*/

/*
  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
  cv::imshow( "Display window", plot_data );                   // Show our image inside it.
  cv::namedWindow( "Display window2", cv::WINDOW_AUTOSIZE );// Create a window for display.
  cv::imshow( "Display window2", plot_result );                   // Show our image inside it.
  cv::waitKey();
*/

}