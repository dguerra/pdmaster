/*
 * MOMFBD.h
 *
 *  Created on: Nov, 2014
 *      Author: dailos
 */

#ifndef MOMFBD_H_
#define MOMFBD_H_
#include <stdio.h>


typedef double float64_t;
typedef unsigned char byte;
typedef float64_t fp_t;  // the default floating point type

struct complex{
  fp_t re;
  fp_t im;
};
typedef struct complex complex_t;

//complex_t **createRandomMatrix(const unsigned int& xSize, const unsigned int& ySize);
complex_t **ct2dim(int x1l,int x1h,int x2l,int x2h);
void del_ct2dim(complex_t **p,int x1l, int x1h, int x2l, int x2h);
void swap(fp_t &a,fp_t &b);
void fft(complex_t *data_in,int nn,int isign);
void fft_n(complex_t *data_in,int *nn,int nd,int isign);
void fft_init(int,int);
void fft_2d(complex_t **&data,int n1,int n2,int isign);
void fft_done(int,int);
void fft_reorder(complex_t **f,int np);

class MOMFBD
{
public:
  bool testMOMFBD();
  
private:
  complex_t **sj;
  fp_t **phi; 
  fp_t **si;                  // windowed image: external
//
// local and constant: to be initialised by image_t
//
  complex_t **dj;                // FT of image: constant but self-initialised
// from image
  fp_t **phi_fixed;           // phi_fixed=PD only? (set to NULL)
  byte restore_obj;
//
// global and constant: to be initialised by imgstack
//
  int np;
// from imgstack
  int nph,nm,offs;            // offs used for windowing
  int *mode_num;              // external data
  fp_t ***psi,**pupil;        //    "      "
  fp_t *atm_rms;              //    "      "; why is this copied for all images?
  fp_t area,nf,lambda;

  void gradient_vogel(fp_t *g,fp_t *alpha,fp_t **q,complex_t **p,fp_t reg_alpha);
};

bool MOMFBD::testMOMFBD()
{
  return true;
}

void MOMFBD::gradient_vogel(fp_t *g,fp_t *alpha,fp_t **q,complex_t **p,fp_t reg_alpha)
{
//  fprintf(stderr,"subimg::subimg: %lX \n",this);
  complex_t **pj=ct2dim(1,np,1,np);
  complex_t **hj=ct2dim(1,np,1,np);
  complex_t **gl=ct2dim(1,np,1,np);
  int ofs=(np-nph)/2;
// in case of severe undersampling
  int ll=1,ul=nph;
  if(ofs<0) ul=(ll-=ofs)+np-1;
//
  memset(pj[1]+1,0,np*np*sizeof(complex_t));
  for(int k=ll;k<=ul;++k)
    for(int l=ll;l<=ul;++l){
      pj[k+ofs][l+ofs].re=pupil[k][l]*cos(phi[k][l]);
      pj[k+ofs][l+ofs].im=pupil[k][l]*sin(phi[k][l]);
    }
  memcpy(hj[1]+1,pj[1]+1,np*np*sizeof(complex_t));
  fft_reorder(hj,np);
  fft_2d(hj,np,np,-1); // hj
  for(int x=1;x<=np;++x)                             // nph>=np/2 ?
    for(int y=1;y<=np;++y){
      complex_t pq={p[x][y].re*q[x][y],p[x][y].im*q[x][y]};
      fp_t ps=p[x][y].re*p[x][y].re+p[x][y].im*p[x][y].im;
      fp_t qs=q[x][y]*q[x][y];
      gl[x][y].re=(pq.re*dj[x][y].re-pq.im*dj[x][y].im-ps*sj[x][y].re)/qs;
      gl[x][y].im=(pq.re*dj[x][y].im+pq.im*dj[x][y].re-ps*sj[x][y].im)/qs;
    }
  fft_reorder(gl,np);
  fft_2d(gl,np,np,-1);                              // real quantity!!! can be done cheaper?
  for(int x=1;x<=np;++x) 
    for(int y=1;y<=np;++y){
      fp_t re=gl[x][y].re;
      gl[x][y].re=hj[x][y].re*re;                   // don't forget to normalise
      gl[x][y].im=hj[x][y].im*re;
    }
  fft_2d(gl,np,np,1);
  fft_reorder(gl,np);
  memset(g+1,0,nm*sizeof(fp_t));
  for(int k=ll;k<=ul;++k){
    int x=k+ofs;
    for(int l=ll;l<=ul;++l){
      int y=l+ofs;
      for(int m=1;m<=nm;++m)
        g[m]-=2.0*(pj[x][y].re*gl[x][y].im-pj[x][y].im*gl[x][y].re)*psi[mode_num[m]][k][l]*pupil[k][l];
    }
  }
  del_ct2dim(gl,1,np,1,np);
  del_ct2dim(hj,1,np,1,np);
  del_ct2dim(pj,1,np,1,np);
  for(int m=1;m<=nm;++m){
    g[m]*=area/(fp_t)(np*np);                   //  ; Kludge:?
    if(reg_alpha) g[m]+=reg_alpha*alpha[m]/atm_rms[m];
  }
}


/*
complex_t **createRandomMatrix(const unsigned int& xSize, const unsigned int& ySize)
{
  double sigma = 5.0;
  std::random_device rdevice;
  std::default_random_engine generator(rdevice());
  std::normal_distribution<> distribution(0, sigma);

  cv::Mat A = cv::Mat(xSize, ySize, cv::DataType<T>::type);
  for(auto it = A.begin<T>(); it != A.end<T>(); ++it)
  {
    (*it) = (T)distribution(generator);
  }
  return A;
}
*/

complex_t **ct2dim(int x1l,int x1h,int x2l,int x2h)
{
  int nx1=x1h-x1l+1,nx2=x2h-x2l+1;
  complex_t **p;
  p = new complex_t* [nx1] - x1l;
  p[x1l]=new complex_t [nx1*nx2] - x2l;
  for(int x1=x1l+1;x1<=x1h;++x1) p[x1]=p[x1-1]+nx2;
  return p;
}

void del_ct2dim(complex_t **p,int x1l, int x1h, int x2l, int x2h)
{
  delete[] (p[x1l]+x2l);
  delete[] (p+x1l);
}

void swap(fp_t &a,fp_t &b)
{
  fp_t c=a;
  a=b;
  b=c;
}

void fft(complex_t *data_in,int nn,int isign)
{
  fp_t *data=(fp_t*)data_in;
  int n=nn<<1,j=1;
  for(int i=1;i<n;i+=2){
    if(j>i){
      swap(data[j],data[i]);
      swap(data[j+1],data[i+1]);
    }
    int m=n>>1;
    while((m>=2)&&(j>m)){
      j-=m;
      m>>=1;
    }
    j+=m;
  }
  int mmax=2;
  while(n>mmax){
    int istep=mmax<<1;
    fp_t theta=isign*(6.28318530717959/mmax);
    fp_t wtemp=sin(0.5*theta);
    fp_t wpr=-2.0*wtemp*wtemp;
    fp_t wpi=sin(theta);
    fp_t wr=1.0;
    fp_t wi=0.0;
    for(int m=1;m<mmax;m+=2){
      for(int i=m;i<=n;i+=istep){
        j=i+mmax;
        fp_t tempr=wr*data[j]-wi*data[j+1];
        fp_t tempi=wr*data[j+1]+wi*data[j];
        data[j]=data[i]-tempr;
        data[j+1]=data[i+1]-tempi;
        data[i]+=tempr;
        data[i+1]+=tempi;
      }
      wr=(wtemp=wr)*wpr-wi*wpi+wr;
      wi=wi*wpr+wtemp*wpi+wi;
    }
    mmax=istep;
  }
}


void fft_n(complex_t *data_in,int *nn,int nd,int isign)
{ 
  fp_t *data=(fp_t*)&(data_in[1])-1;
  int ip1,ip2,ip3,nt=1;
  for(int id=1;id<=nd;++id) nt*=nn[id];
  int nprev=1;
  for(int id=nd;id>=1;--id){
    int n=nn[id];
    int nrem=nt/(n*nprev);
    ip1=nprev<<1;
    ip2=ip1*n;
    ip3=ip2*nrem;
    int i2rev=1;
    for(int i2=1;i2<=ip2;i2+=ip1){
      if(i2<i2rev)
	for(int i1=i2;i1<=i2+ip1-2;i1+=2)
	  for(int i3=i1;i3<=ip3;i3+=ip2){
            int i3rev=i2rev+i3-i2;
	    swap(data[i3],data[i3rev]);
	    swap(data[i3+1],data[i3rev+1]);
	  }
      int ibit=ip2>>1;
      while(ibit>=ip1&&i2rev>ibit){
	i2rev-=ibit;
	ibit>>=1;
      }
      i2rev+=ibit;
    }
    int ifp1=ip1;
    while(ifp1<ip2){
      int ifp2=ifp1<<1;
      fp_t theta=isign*6.28318530717959/(ifp2/ip1);
      fp_t wtemp=sin(0.5*theta);
      fp_t wpr= -2.0*wtemp*wtemp;
      fp_t wpi=sin(theta);
      fp_t wr=1.0;
      fp_t wi=0.0;
      for(int i3=1;i3<=ifp1;i3+=ip1){
	for(int i1=i3;i1<=i3+ip1-2;i1+=2)
	  for(int i2=i1;i2<=ip3;i2+=ifp2){
	    int k1=i2;
	    int k2=k1+ifp1;
	    fp_t tempr=(fp_t)wr*data[k2]-(fp_t)wi*data[k2+1];
	    fp_t tempi=(fp_t)wr*data[k2+1]+(fp_t)wi*data[k2];
	    data[k2]=data[k1]-tempr;
	    data[k2+1]=data[k1+1]-tempi;
	    data[k1]+=tempr;
	    data[k1+1]+=tempi;
	  }
	wr=(wtemp=wr)*wpr-wi*wpi+wr;
	wi=wi*wpr+wtemp*wpi+wi;
      }
      ifp1=ifp2;
    }
    nprev*=n;
  }
}

void fft_init(int,int)
{
}

void fft_2d(complex_t **&data,int n1,int n2,int isign)
{
  int nn[3]={0,n1,n2};
  fft_n(data[1],nn,2,isign);
  if(isign<0){
    fp_t n=(fp_t)(n1*n2);
    for(int x1=1;x1<=n1;++x1)
      for(int x2=1;x2<=n2;++x2){
        data[x1][x2].re/=n;
        data[x1][x2].im/=n;
      }
  }
}

void fft_done(int,int)
{
}

void fft_reorder(complex_t **f,int np)
{
  int nh=np/2;
  complex_t *buf=new complex_t [nh];
  for(int x=1;x<=nh;++x){
    memcpy(buf,f[x]+1,nh*sizeof(complex_t));
    memcpy(f[x]+1,f[x+nh]+nh+1,nh*sizeof(complex_t));
    memcpy(f[x+nh]+nh+1,buf,nh*sizeof(complex_t));
//
    memcpy(buf,f[x+nh]+1,nh*sizeof(complex_t));
    memcpy(f[x+nh]+1,f[x]+nh+1,nh*sizeof(complex_t));
    memcpy(f[x]+nh+1,buf,nh*sizeof(complex_t));
  }
  delete[] buf;
}


#endif /* MOMFBD_H_ */