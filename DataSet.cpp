/*
 * DataSet.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: dailos
 */

//http://www.cs.toronto.edu/~kriz/cifar.html

#include "DataSet.h"
#include <fstream>
int DataSet::single_color_image()
{
  //Read ground truth image from fits file
  cv::Mat img;

  img = cv::imread(std::string("../inputs/lena30.jpg"), CV_LOAD_IMAGE_COLOR);   // Read the file

  if(! img.data )                              // Check for invalid input
  {
    std::cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }
  
  img.convertTo(img, CV_32FC3);

  unsigned int origen = 250;
  unsigned int tileSize = 32;
  img = img(cv::Range(origen, origen + tileSize), cv::Range(origen, origen + tileSize)).clone();
  //Opencv follow BGR order
  // "channels" is a vector of 3 Mat arrays:
  std::vector<cv::Mat> channels;
  cv::split(img, channels);
  std::ofstream out("../image.dat", std::ios::out | std::ios::binary);
  if(!out) {
    std::cout << "Cannot open file.";
    return -1;
  }
  
  //However cifar10 data set are stored in order RGB
  cv::Mat label = cv::Mat::ones(10,1,cv::DataType<float>::type);
  out.write(reinterpret_cast<char*>(label.data), label.total() * label.elemSize());
  out.write(reinterpret_cast<char*>(channels.at(2).data), channels.at(2).total() * channels.at(2).elemSize());
  out.write(reinterpret_cast<char*>(channels.at(1).data), channels.at(1).total() * channels.at(1).elemSize());
  out.write(reinterpret_cast<char*>(channels.at(0).data), channels.at(0).total() * channels.at(0).elemSize());


  out.close();
 
  char arr_l[10*4];
  char arr_b[32*32*4];
  char arr_g[32*32*4];
  char arr_r[32*32*4];
  
  std::ifstream in("../image.dat", std::ios::in | std::ios::binary);
  in.read(reinterpret_cast<char*>(arr_l), 10*4);
  in.read(reinterpret_cast<char*>(arr_r), 32*32*4);
  in.read(reinterpret_cast<char*>(arr_g), 32*32*4);
  in.read(reinterpret_cast<char*>(arr_b), 32*32*4);

  // see how many bytes have been read
  std::cout << in.gcount() << " bytes read\n";

  in.close();
  cv::Mat img_out_l(10, 1, cv::DataType<float>::type, arr_l);
  std::cout << "label=" << img_out_l << std::endl; 
  cv::Mat img_out_r(img.size(), img.depth(), arr_r);
  cv::Mat img_out_g(img.size(), img.depth(), arr_g);
  cv::Mat img_out_b(img.size(), img.depth(), arr_b);
  std::vector<cv::Mat> img_mix = {img_out_b, img_out_g, img_out_r};
  cv::Mat img_out;
  cv::merge(img_mix, img_out);
  cv::normalize(img_out, img_out, 0.0, 255.0, CV_MINMAX);
  cv::imwrite("../img_out_dataset.jpg", img_out);
  return 0;
}

DataSet::DataSet()
{
  // TODO default constructor
}


DataSet::~DataSet()
{
  // TODO Auto-generated destructor stub
}
