/*
 * ToolBox.h
 *
 *  Created on: Oct 25, 2013
 *      Author: dailos
 */

#ifndef TOOLBOX_H_
#define TOOLBOX_H_
#include <iostream>
#include "opencv2/opencv.hpp"

void divideIntoTiles(const cv::Size& dim, const unsigned int& pixelsBetweenTiles, const unsigned int& tileSize, std::vector<std::pair<cv::Range,cv::Range> >& tileRngs);
void shuffleRows(const cv::Mat &matrix, cv::Mat& shuffleMatrix);
double rms(const cv::Mat& A, const cv::Mat& mask = cv::Mat());
void cholesky(const cv::Mat& AA, cv::Mat& LL);
void show_matrix(double *A, int n);
void reorderColumns(const cv::Mat& A, const unsigned int& slice, cv::Mat& Ar);
cv::Mat conjComplex(const cv::Mat& A);
void partlyKnownDifferencesInPhaseConstraints(int M, int K, cv::Mat& Q2);
void householder(const cv::Mat &m, cv::Mat &Q, cv::Mat &R);
void shift(cv::Mat& I, cv::Mat& O, const int& cx, const int& cy);
cv::Mat crosscorrelation(const cv::Mat& A, const cv::Mat& B);
cv::Mat crosscorrelation_direct(const cv::Mat& A, const cv::Mat& B);
void convolve(const cv::Mat& imgOriginal, const cv::Mat& kernel, cv::Mat& out, const bool& corr = false, const bool& full = false);
void conv_flaw(const cv::Mat& imgOriginal, const cv::Mat& kernel, cv::Mat& out, const bool& corr = false);
void convolveDFT(const cv::Mat& imgOriginal, const cv::Mat& kernel, cv::Mat& out, const bool& corr = false, const bool& full = false);
unsigned int optimumSideLength(const unsigned int& minimumLength, const double& radiousLength);
cv::Mat takeoutImageCore(const cv::Mat& im, const unsigned int& imageCoreSize);
cv::Mat selectCentralROI(const cv::Mat& im, const cv::Size& roiSize);
cv::Mat centralROI(const cv::Mat& im, const cv::Size& roiSize, cv::Mat& roi);
//void writeOnImage(cv::Mat& img, const std::string& text);
//cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows);
cv::Mat absComplex(const cv::Mat& complexI);
cv::Mat normComplex(const cv::Mat& A, cv::Mat& out);
cv::Mat divComplex(const cv::Mat& A, const cv::Mat& B);
//void shiftQuadrants(cv::Mat& I);
//void showComplex(const cv::Mat& A, const std::string& txt, const bool& shiftQ = true, const bool& logScale = false);
//void showHistogram(const cv::Mat& src);
std::pair<cv::Mat, cv::Mat> splitComplex(const cv::Mat& I);
cv::Mat makeComplex(const cv::Mat& real, const cv::Mat& imag);
cv::Mat makeComplex(const cv::Mat& real);
long factorial(const long& theNumber);


//methods from phasecorr.cpp
void fftShift(cv::Mat& out);
void divSpectrums(const cv::Mat& srcA, const cv::Mat& srcB, cv::Mat& dst, int flags, bool conjB = false);
void magSpectrums(const cv::Mat& src, cv::Mat& dst);

class Plot2d
{
    public:

    Plot2d(cv::InputArray plotData)
    {
        cv::Mat _plotData = plotData.getMat();
        //if the matrix is not Nx1 or 1xN
        if(_plotData.cols > 1 && _plotData.rows > 1)
        {
            std::cout << "ERROR: Plot data must be a 1xN or Nx1 matrix." << std::endl;
            exit(0);
        }

        //if the matrix type is not CV_64F
        if(_plotData.type() != CV_64F)
        {
            std::cout << "ERROR: Plot data type must be double (CV_64F)." << std::endl;
            exit(0);
        }

        //in case we have a row matrix than needs to be transposed
        if(_plotData.cols > _plotData.rows)
        {
            _plotData = _plotData.t();
        }

        plotDataY=_plotData;
        plotDataX = plotDataY*0;
        for (int i=0; i<plotDataY.rows; i++)
        {
            plotDataX.at<double>(i,0) = i;
        }

        //calling the main constructor
        plotHelper(plotDataX, plotDataY);

    }

    Plot2d(cv::InputArray plotDataX_, cv::InputArray plotDataY_)
    {
        cv::Mat _plotDataX = plotDataX_.getMat();
        cv::Mat _plotDataY = plotDataY_.getMat();
        //f the matrix is not Nx1 or 1xN
        if((_plotDataX.cols > 1 && _plotDataX.rows > 1) || (_plotDataY.cols > 1 && _plotDataY.rows > 1))
        {
            std::cout << "ERROR: Plot data must be a 1xN or Nx1 matrix." << std::endl;
            exit(0);
        }

        //if the matrix type is not CV_64F
        if(_plotDataX.type() != CV_64F || _plotDataY.type() != CV_64F)
        {
            std::cout << "ERROR: Plot data type must be double (CV_64F)." << std::endl;
           exit(0);
        }

        //in case we have a row matrix than needs to be transposed
        if(_plotDataX.cols > _plotDataX.rows)
        {
            _plotDataX = _plotDataX.t();
        }
        if(_plotDataY.cols > _plotDataY.rows)
        {
            _plotDataY = _plotDataY.t();
        }

        plotHelper(_plotDataX, _plotDataY);
    }

    //set functions
    void setMinX(double _plotMinX)
    {
        plotMinX = _plotMinX;
        plotMinX_plusZero = _plotMinX;
    }
    void setMaxX(double _plotMaxX)
    {
        plotMaxX = _plotMaxX;
        plotMaxX_plusZero = _plotMaxX;
    }
    void setMinY(double _plotMinY)
    {
        plotMinY = _plotMinY;
        plotMinY_plusZero = _plotMinY;
    }
    void setMaxY(double _plotMaxY)
    {
        plotMaxY = _plotMaxY;
        plotMaxY_plusZero = _plotMaxY;
    }
    void setPlotLineWidth(int _plotLineWidth)
    {
        plotLineWidth = _plotLineWidth;
    }
    void setNeedPlotLine(bool _needPlotLine)
    {
        needPlotLine = _needPlotLine;
    }
    void setPlotLineColor(cv::Scalar _plotLineColor)
    {
        plotLineColor=_plotLineColor;
    }
    void setPlotBackgroundColor(cv::Scalar _plotBackgroundColor)
    {
        plotBackgroundColor=_plotBackgroundColor;
    }
    void setPlotAxisColor(cv::Scalar _plotAxisColor)
    {
        plotAxisColor=_plotAxisColor;
    }
    void setPlotGridColor(cv::Scalar _plotGridColor)
    {
        plotGridColor=_plotGridColor;
    }
    void setPlotTextColor(cv::Scalar _plotTextColor)
    {
        plotTextColor=_plotTextColor;
    }
    void setPlotSize(int _plotSizeWidth, int _plotSizeHeight)
    {
        if(_plotSizeWidth > 400)
            plotSizeWidth = _plotSizeWidth;
        else
            plotSizeWidth = 400;

        if(_plotSizeHeight > 300)
            plotSizeHeight = _plotSizeHeight;
        else
            plotSizeHeight = 300;
    }

    //render the plotResult to a Mat
    void render(cv::OutputArray _plotResult)
    {
        //create the plot result
        _plotResult.create(plotSizeHeight, plotSizeWidth, CV_8UC3);
        plotResult = _plotResult.getMat();
        plotResult.setTo(plotBackgroundColor);

        int NumVecElements = plotDataX.rows;

        cv::Mat InterpXdata = linearInterpolation(plotMinX, plotMaxX, 0, plotSizeWidth, plotDataX);
        cv::Mat InterpYdata = linearInterpolation(plotMinY, plotMaxY, 0, plotSizeHeight, plotDataY);

        //Find the zeros in image coordinates
        cv::Mat InterpXdataFindZero = linearInterpolation(plotMinX_plusZero, plotMaxX_plusZero, 0, plotSizeWidth, plotDataX_plusZero);
        cv::Mat InterpYdataFindZero = linearInterpolation(plotMinY_plusZero, plotMaxY_plusZero, 0, plotSizeHeight, plotDataY_plusZero);

        int ImageXzero = (int)InterpXdataFindZero.at<double>(NumVecElements,0);
        int ImageYzero = (int)InterpYdataFindZero.at<double>(NumVecElements,0);

        double CurrentX = plotDataX.at<double>(NumVecElements-1,0);
        double CurrentY = plotDataY.at<double>(NumVecElements-1,0);

        drawAxis(ImageXzero,ImageYzero, CurrentX, CurrentY, plotAxisColor, plotGridColor);

        if(needPlotLine)
        {
            //Draw the plot by connecting lines between the points
            cv::Point p1;
            p1.x = (int)InterpXdata.at<double>(0,0);
            p1.y = (int)InterpYdata.at<double>(0,0);

            for (int r=1; r<InterpXdata.rows; r++)
            {
                cv::Point p2;
                p2.x = (int)InterpXdata.at<double>(r,0);
                p2.y = (int)InterpYdata.at<double>(r,0);

                cv::line(plotResult, p1, p2, plotLineColor, plotLineWidth, 8, 0);

                p1 = p2;
            }
        }
        else
        {
            for (int r=0; r<InterpXdata.rows; r++)
            {
                cv::Point p;
                p.x = (int)InterpXdata.at<double>(r,0);
                p.y = (int)InterpYdata.at<double>(r,0);

                cv::circle(plotResult, p, 1, plotLineColor, plotLineWidth, 8, 0);
            }
        }
    }

    protected:

    cv::Mat plotDataX;
    cv::Mat plotDataY;
    cv::Mat plotDataX_plusZero;
    cv::Mat plotDataY_plusZero;
    const char * plotName;

    //dimensions and limits of the plot
    int plotSizeWidth;
    int plotSizeHeight;
    double plotMinX;
    double plotMaxX;
    double plotMinY;
    double plotMaxY;
    double plotMinX_plusZero;
    double plotMaxX_plusZero;
    double plotMinY_plusZero;
    double plotMaxY_plusZero;
    int plotLineWidth;

    //colors of each plot element
    cv::Scalar plotLineColor;
    cv::Scalar plotBackgroundColor;
    cv::Scalar plotAxisColor;
    cv::Scalar plotGridColor;
    cv::Scalar plotTextColor;

    //the final plot result
    cv::Mat plotResult;

    //flag which enables/disables connection of plotted points by lines
    bool needPlotLine;

    void plotHelper(cv::Mat _plotDataX, cv::Mat _plotDataY)
    {
        plotDataX=_plotDataX;
        plotDataY=_plotDataY;

        int NumVecElements = plotDataX.rows;

        plotDataX_plusZero = cv::Mat::zeros(NumVecElements+1,1,CV_64F);
        plotDataY_plusZero = cv::Mat::zeros(NumVecElements+1,1,CV_64F);

        for(int i=0; i<NumVecElements; i++){
            plotDataX_plusZero.at<double>(i,0) = plotDataX.at<double>(i,0);
            plotDataY_plusZero.at<double>(i,0) = plotDataY.at<double>(i,0);
        }

        double MinX;
        double MaxX;
        double MinY;
        double MaxY;
        double MinX_plusZero;
        double MaxX_plusZero;
        double MinY_plusZero;
        double MaxY_plusZero;

        needPlotLine = true;

        //Obtain the minimum and maximum values of Xdata
        cv::minMaxLoc(plotDataX,&MinX,&MaxX);

        //Obtain the minimum and maximum values of Ydata
        cv::minMaxLoc(plotDataY,&MinY,&MaxY);

        //Obtain the minimum and maximum values of Xdata plus zero
        cv::minMaxLoc(plotDataX_plusZero,&MinX_plusZero,&MaxX_plusZero);

        //Obtain the minimum and maximum values of Ydata plus zero
        cv::minMaxLoc(plotDataY_plusZero,&MinY_plusZero,&MaxY_plusZero);

        //setting the min and max values for each axis
        plotMinX = MinX;
        plotMaxX = MaxX;
        plotMinY = MinY;
        plotMaxY = MaxY;
        plotMinX_plusZero = MinX_plusZero;
        plotMaxX_plusZero = MaxX_plusZero;
        plotMinY_plusZero = MinY_plusZero;
        plotMaxY_plusZero = MaxY_plusZero;

        //setting the default size of a plot figure
        setPlotSize(600, 400);

        //setting the default plot line size
        setPlotLineWidth(1);

        //setting default colors for the different elements of the plot
        setPlotAxisColor(cv::Scalar(0, 0, 255));
        setPlotGridColor(cv::Scalar(255, 255, 255));
        setPlotBackgroundColor(cv::Scalar(0, 0, 0));
        setPlotLineColor(cv::Scalar(0, 255, 255));
        setPlotTextColor(cv::Scalar(255, 255, 255));
    }

    void drawAxis(int ImageXzero, int ImageYzero, double CurrentX, double CurrentY, cv::Scalar axisColor, cv::Scalar gridColor)
    {
        drawValuesAsText(0, ImageXzero, ImageYzero, 10, 20);
        drawValuesAsText(0, ImageXzero, ImageYzero, -20, 20);
        drawValuesAsText(0, ImageXzero, ImageYzero, 10, -10);
        drawValuesAsText(0, ImageXzero, ImageYzero, -20, -10);
        drawValuesAsText("X = %g",CurrentX, 0, 0, 40, 20);
        drawValuesAsText("Y = %g",CurrentY, 0, 20, 40, 20);

        //Horizontal X axis and equispaced horizontal lines
        int LineSpace = 50;
        int TraceSize = 5;
        drawLine(0, plotSizeWidth, ImageYzero, ImageYzero, axisColor);

       for(int i=-plotSizeHeight; i<plotSizeHeight; i=i+LineSpace){

            if(i!=0){
                int Trace=0;
                while(Trace<plotSizeWidth){
                    drawLine(Trace, Trace+TraceSize, ImageYzero+i, ImageYzero+i, gridColor);
                    Trace = Trace+2*TraceSize;
                }
            }
        }


        //Vertical Y axis
        drawLine(ImageXzero, ImageXzero, 0, plotSizeHeight, axisColor);

        for(int i=-plotSizeWidth; i<plotSizeWidth; i=i+LineSpace){

            if(i!=0){
                int Trace=0;
                while(Trace<plotSizeHeight){
                    drawLine(ImageXzero+i, ImageXzero+i, Trace, Trace+TraceSize, gridColor);
                    Trace = Trace+2*TraceSize;
                }
            }
        }
    }

    cv::Mat linearInterpolation(double Xa, double Xb, double Ya, double Yb, cv::Mat Xdata){

        cv::Mat Ydata = Xdata*0;

        for (int i=0; i<Xdata.rows; i++){

            double X = Xdata.at<double>(i,0);
            Ydata.at<double>(i,0) = int(Ya + (Yb-Ya)*(X-Xa)/(Xb-Xa));

            if(Ydata.at<double>(i,0)<0)
                Ydata.at<double>(i,0)=0;

        }

        return Ydata;
    }

    void drawValuesAsText(double Value, int Xloc, int Yloc, int XMargin, int YMargin){

        char AxisX_Min_Text[20];
        double TextSize = 1;

        sprintf(AxisX_Min_Text, "%g", Value);
        cv::Point AxisX_Min_Loc;
        AxisX_Min_Loc.x = Xloc+XMargin;
        AxisX_Min_Loc.y = Yloc+YMargin;

        putText(plotResult,AxisX_Min_Text, AxisX_Min_Loc, cv::FONT_HERSHEY_COMPLEX_SMALL, TextSize, plotTextColor, 1, 8);
    }

    void drawValuesAsText(const char *Text, double Value, int Xloc, int Yloc, int XMargin, int YMargin){

        char AxisX_Min_Text[20];
        int TextSize = 1;

        sprintf(AxisX_Min_Text, Text, Value);
        cv::Point AxisX_Min_Loc;
        AxisX_Min_Loc.x = Xloc+XMargin;
        AxisX_Min_Loc.y = Yloc+YMargin;

        putText(plotResult,AxisX_Min_Text, AxisX_Min_Loc, cv::FONT_HERSHEY_COMPLEX_SMALL, TextSize, plotTextColor, 1, 8);
    }


    void drawLine(int Xstart, int Xend, int Ystart, int Yend, cv::Scalar lineColor){

        cv::Point Axis_start;
        cv::Point Axis_end;
        Axis_start.x = Xstart;
        Axis_start.y = Ystart;
        Axis_end.x = Xend;
        Axis_end.y = Yend;

        cv::line(plotResult, Axis_start, Axis_end, lineColor, plotLineWidth, 8, 0);
    }

};


#endif /* TOOLBOX_H_ */
