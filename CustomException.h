/*
 * CustomException.h
 *
 *  Created on: Oct 16, 2013
 *      Author: dailos
 */

#ifndef CUSTOMEXCEPTION_H_
#define CUSTOMEXCEPTION_H_

#include <iostream>
#include <stdexcept>

//Simple implementation of exception class.
//custom messages added by objects
class CustomException : public std::exception
{
 public:
  CustomException(std::string str) : m_str(str) {}
  virtual const char * what() const throw () {return m_str.c_str(); }
  ~CustomException() throw() {}
 protected:
  std::string m_str;
};

#endif /* CUSTOMEXCEPTION_H_ */
