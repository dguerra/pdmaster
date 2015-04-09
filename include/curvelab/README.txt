Go to www.fftw.org download library 2.1.5:
Unzip, go to folder and first './configure' and then 'make'

visit www.curvelets.org
Download library: make

Go to foder fdct_wrapping_cpp and build shared library manually:
g++ -Wall -shared -fPIC -o libfdct_wrapping.so fdct_wrapping.o  fdct_wrapping_param.o  ifdct_wrapping.o

Go to foder fdct_usfft_cpp and build shared library manually:
g++ -Wall -shared -fPIC -o libfdct_usfft.so afdct_usfft.o  fdct_usfft.o  fdct_usfft_param.o  ifdct_usfft.o