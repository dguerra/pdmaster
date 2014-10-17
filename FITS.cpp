/*
 * FITS.cpp
 *
 *  Created on: Feb 28, 2014
 *      Author: dailos
 */

#include "FITS.h"
#include "fitsio.h"
#include "CustomException.h"

//Known bug.!: When reading an image some flawed pixel are added around, so the image needs to be cropped afterwards in order to avoid them

void readFITS(const std::string& fitsname, cv::Mat& cvImage)
{

  fitsfile *fptr = nullptr;
  int status(0);
  char err_text[100];
  //READONLY, READWRITE

  fits_open_file(&fptr, fitsname.c_str(), READONLY, &status);
  if (status)
  {
    fits_report_error(stdout, status);
    fits_get_errstatus(status,err_text);
    fptr = nullptr;
    std::cout << "readFITS: Unable to open the fits file." << std::endl;
    throw  CustomException("readFITS: Unable to open the fits file.");
  }

  //turn off scaling so we read the raw pixel value
  double bscale_ = 1.0, bzero_ = 0.0;
  fits_set_bscale(fptr,  bscale_, bzero_, &status);
  int bitpix, naxis;
  int maxdim(2);
  long naxes[] = {1, 1};

  fits_get_img_param(fptr, maxdim,  &bitpix, &naxis, naxes, &status);
  if (status)
  {
    fits_report_error(stdout, status);
    fits_get_errstatus(status,err_text);
    fptr = nullptr;
    std::cout << "readFITS: Unable to get params from FITS." << std::endl;
    throw  CustomException("readFITS: Unable to get params from FITS.");
  }
  if(naxis != 2)
  {
    throw CustomException("readFITS: Wrong number of dimensions in FITS file. Only two dims image supported.");
  }

  //  TBYTE, TSBYTE, TSHORT, TUSHORT, TINT, TUINT, TLONG, TLONGLONG, TULONG, TFLOAT, TDOUBLE
  long fpixel[] = {1, 1};
  long lpixel[] = {naxes[0], naxes[1]};
  long inc[] = {1, 1};
  long nelements = naxes[0] * naxes[1];
  double *array = new double[nelements];

  fits_read_subset(fptr, TDOUBLE, fpixel, lpixel, inc, nullptr,  array, nullptr, &status);
  if (status)
  {
    fits_report_error(stdout, status);
    fits_get_errstatus(status,err_text);
    fptr = nullptr;
    delete[] array;
    throw  CustomException("readFITS: Unable to read the fits file.");
  }

  //it seems cfitsio interprets image axes in the oppsite way of opencv
  cvImage = cv::Mat(naxes[1], naxes[0], cv::DataType<double>::type, array);

  fits_close_file(fptr, &status);
  if (status)
  {
    fits_report_error(stdout, status);
    fits_get_errstatus(status,err_text);
    fptr = nullptr;
    delete[] array;
    throw  CustomException("readFITS: Cannot close fits file.");
  }
}


void writeFITS(const cv::Mat& cvImage, const std::string& filename)
{
  fitsfile *fptr; //pointer to the FITS file defined in fitsioh
  int status;
  char err_text[100];
  cv::Mat floatImg = cv::Mat_<float>(cvImage);
  long fpixel = 1, naxis = 2, nelements;
  long naxes[2] = {cvImage.cols, cvImage.rows};
  float array[cvImage.cols][cvImage.rows];
  for(int i = 0;i < floatImg.rows ;i++)
  {
    for(int j = 0;j < floatImg.cols ;j++)
    {
      array[j][i] = floatImg.at<float>(j,i);
    }
  }

  status=0;
  fits_create_file(&fptr, filename.c_str(), &status);
  if (status)
  {
    fits_report_error(stdout, status);
    fits_get_errstatus(status,err_text);
    fptr = nullptr;
    throw  CustomException("writeFITS: Cannot create fits file.");
  }

  fits_create_img(fptr, FLOAT_IMG, naxis, naxes, &status);
  if (status)
  {
    fits_report_error(stdout, status);
    fits_get_errstatus(status,err_text);
    fptr = nullptr;
    throw  CustomException("writeFITS: Cannot create image file.");
  }
  nelements = naxes[0] * naxes[1];
  fits_write_img(fptr, TFLOAT, fpixel, nelements, array[0],  &status);
  if (status)
  {
    fits_report_error(stdout, status);
    fits_get_errstatus(status,err_text);
    fptr = nullptr;
    throw  CustomException("writeFITS: Cannot write image file.");
  }
  fits_close_file(fptr, &status);
  fits_report_error(stderr, status);

}
