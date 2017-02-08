/*
 * DataSet.h
 *
 *  Created on: Noc 11, 2016
 *      Author: dailos
 */

#ifndef DATASET_H_
#define DATASET_H_
#include "opencv2/opencv.hpp"
//Rename module as simply "Optics"

class DataSet
{
public:
  DataSet();
  virtual ~DataSet();
  int single_color_image();
  private:
};

#endif /* DATASET_H_ */