#include <iostream>
#include "TF1.h"
#include "stdio.h"

/*
 Return current for a given magnetic field 
 for Hall A Moller Polarimeter
 based on Sasha's parameterization 
*/

double get_current(double field);
double f_mag(double* x, double *par);

int magnet;

double kg2a(double field, int mag, bool print_out = true)
{

  if(mag < 1 || mag > 5)
    {
      cout << "ERROR: magnet number out of range" << endl;
      return -1;
    }

  magnet = mag;

  double current = get_current(field);
  if(print_out){
    cout << "Field (kG): " << field << " Current (A): " << current << endl;
  }
  return current;
}

double get_current(double field)
{
  
  const int NMAX = 1000; // Max number of iterations
  double err = 1.e-9; // tolerance

  double x;
  double xnew = 1; // initial starting point
  double func;
  double dfunc;

  TF1* f1 = new TF1("f1", f_mag, 0, 500., 0);
  // f1->Draw();

  for(int iter=0; iter<NMAX; iter++)
    {
      x = xnew;
      func = field - f1->Eval(x);
      dfunc = f1->Derivative(x);

      xnew = x + (func/dfunc);
      if(fabs(x-xnew) < err)
        {
          return xnew;
        }
      else if( (iter == NMAX -1) && (fabs(x-xnew) > err) )
        {
          cout << "not converged " << x << endl;
        }
    }

  return 0;

}

double get_field(double current, int mag)
{
  magnet = mag;
  TF1* f1 = new TF1("f1", f_mag, 0, 500., 0);
  return f1->Eval(current);
}

double f_mag(double* x, double* par)
{
  double current = x[0];
  double cn = current / 300.;

  double gl1 = 0;
  double gl2 = 0;
  double gln = 0;
  double fld = 0;

  if(magnet == 1)  
    {
      // Moller Quad Q1/MQO1H01/LARGE/new/white
      gl1 = (.0110605+5.33237*cn-.0142794*pow(cn,2)+.259313*pow(cn,3));
      gln = (gl1+0.0058174*pow(cn,4)-0.831887*pow(cn,5));
      fld = gln*10.*5.08/36.5723;
    }
  else if(magnet == 2)
    {
      // Moller Quad Q2/PATSY/MQM1H02/SMALL/RED
      gl1=(0.0196438+5.35443*cn+0.0297273*pow(cn,2)+0.103505*pow(cn,3));
      gln=(gl1-0.0449275*pow(cn,4)-0.211868*pow(cn,5));
      fld=gln*10.*5.08/44.76;
    }
  else if(magnet == 3)
    {
      // Moller Quad Q3/TESSA/MQO1H03/LARGE/BLUE
      gl1=(0.000632446+5.15178*cn-0.00262778*pow(cn,2));
      gl2=(-0.107635*pow(cn,3)+0.00209902*pow(cn,4));
      gln=(gl1+gl2-0.640635*pow(cn,5));
      fld=gln*10.*5.08/36.74 ;
    }
  else if(magnet == 4)
    {
      // Moller Quad Q4/FELICIA/MQO1H03A/LARGE/BLUE
      gl1=(0.0001732+5.2119*cn-0.000732518*pow(cn,2));
      gl2=(-0.133423*pow(cn,3)+0.000618402*pow(cn,4));
      gln=(gl1+gl2-0.647082*pow(cn,5));
      fld=gln*10.*5.08/36.50;
    }
  else if(magnet == 5)
    {
      // Moller Dipole LILLY/MMA1H01/Blue
      fld=(-0.39026E-04+0.027051*current-0.17799E-08*pow(current,2));
    }

  return fld;

}

