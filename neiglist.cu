/*
 * DEM-GPU code modified by Jieqing Gan 
 * from Nivdia SDK code sample particle,
 * and based on the DEM packing code for ellipsoidal particle 
 * coded by zongyan zhou in FORTRAN. 
 */

/* 
 * Device code.
 */

#ifndef _NEIGLIST_H_
#define _NEIGLIST_H_

#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "math_functions.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_11_atomic_functions.h"

//#include "sm_20_atomic_functions.h"

__constant__ double cua=1.0;
__constant__ double cub=1.55;
__constant__ double xsmall=1e-30;
__constant__ double gg=9.81;

struct Matproperties {
       double estarp;
	double ufricp;
       double vpois;
       double dampnp;
       double damptp;
       double upp;
        };
struct Wallproperties{
       double estarw;
	double ufricw;
       double vpoisw;
       double dampnw;
       double damptw;
       double upw;
       };
__constant__ Matproperties Par[1];
__constant__ Wallproperties ParW[1];

typedef unsigned int uint;

// calculate position in uniform grid
__device__ int3 calcGridPos(double3 p,
                            double3 worldOrigin,
                            double3 cellSize
                            )
{
    int3 gridPos;
    gridPos.x = floor((p.x - worldOrigin.x) / cellSize.x);
    gridPos.y = floor((p.y - worldOrigin.y) / cellSize.y);
    gridPos.z = floor((p.z - worldOrigin.z) / cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridAddress(int3 gridPos,
                                uint3 gridSize)
{
    gridPos.x = max(0, min(gridPos.x, gridSize.x-1));
    gridPos.y = max(0, min(gridPos.y, gridSize.y-1));
    gridPos.z = max(0, min(gridPos.z, gridSize.z-1));
    return  gridPos.z*gridSize.y*gridSize.x + gridPos.y*gridSize.x + gridPos.x;
}

// calculate grid hash value for each particle
__global__ void calcHashD(uint numBodies,
          double3* pos,
          uint2*  particleHash,
          uint3   gridSize,
          double3  cellSize,
          double3  worldOrigin
          )
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index<numBodies)
   {
    double3 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(p, worldOrigin, cellSize);
    uint gridAddress = calcGridAddress(gridPos, gridSize);

   // store grid hash and particle index
    particleHash[index] = make_uint2(gridAddress, index);
  }
}


// find start of each cell in sorted particle list by comparing with previous hash value
// one thread per particle
__global__ void findCellStartD(uint2* particleHash,
               uint* cellStart,
               uint   numBodies,
               uint   numCells)
{
    uint i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i<numBodies)
   {
    uint cell = particleHash[i].x;

    if (i > 0) {
        if (cell != particleHash[i-1].x) cellStart[cell] = i;
    } 
	else {
        cellStart[cell] = i;
    }
  }
}

__global__ void addTwoArraysD(uint *b, uint *a, uint numBodies)

{
  uint i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<numBodies)
   {
    b[i] += a[i];
   }
}

//-----------------------------------------------------------------
__device__ void vprojec(double3 vecp,double3 vecu,double3 *vecr,uint ityp)
       { 
        double3 vecq;
	double pmag,rmag,pr; 
        vecq=cross(vecu,vecp);
	*vecr=cross(-vecu,vecq);
       if(ityp==0)return;
	   pmag=length(vecp);
       if(pmag<0.001)return;
       rmag=length(*vecr);
       if(pmag>rmag*100)return;
       pr=sqrt(pmag/rmag);
       *vecr=*vecr*pr;
       }
//-----------------------------------------------------------------
__device__ void vscale(double3 *vec,double p1,double p2)
//    vector scaled 
      {double ratio;
       ratio=p1/p2;
       *vec=*vec*ratio;
       }


__global__ void calculateforcepairD(int totalcontacts,
                   double3* pos,
	               double3* pdis,
	               double3* angv,
	               double* rad,
	               double* rmass,

                   uint* Lp,
                   uint* Ln,
                   uint* oldLn,
                   double3 *dispt,
                   double *fricp,
                   double3 *olddispt,
                   double *oldfricp,
                   double  dt,
                   // return value
                   double3* forcepair,
                   double3* torqpairi,
		     double3* torqpairj,
                   uint* contactpair)
{
uint icontacti;
double3 olddispti,dispti,fi,ti,tj;
double oldfricpi,fricpi;
//parallerized loop over Ilist
int Ilist = blockIdx.x*blockDim.x + threadIdx.x;  

if(Ilist<totalcontacts)
{
uint i=Lp[Ilist];
uint j=Ln[Ilist];

icontacti=0;

 double3 posi = pos[i];
 double3 pdisi = pdis[i];
 double3  angvi = angv[i];
 double radi = rad[i];
 double rmassi=rmass[i];


 double3 posj = pos[j];
 double3 pdisj = pdis[j];
 double3  angvj = angv[j];
 double radj = rad[j];
 double rmassj=rmass[j];


 dispti=dispt[Ilist];
 fricpi=fricp[Ilist];

// cauculate force and torque for each contact particle pair	
 calforceij(posi,posj,pdisi,pdisj,angvi,angvj,radi,radj,rmassi,rmassj,dispti,fricpi,dt,
		&olddispti,&oldfricpi,&icontacti,&fi,&ti,&tj);

 if(icontacti==1)
 {
 forcepair[Ilist] = fi;
 torqpairi[Ilist] =ti;
 torqpairj[Ilist] =tj;
 dispt[Ilist]=olddispti;
 fricp[Ilist]=oldfricpi;
 contactpair[Ilist]=1;
 }
 else{
 forcepair[Ilist] = make_double3(0.0f);
 torqpairi[Ilist] =make_double3(0.0f);
 torqpairj[Ilist] =make_double3(0.0f);
 dispt[Ilist]=make_double3(0.0f);
 fricp[Ilist]=0.0;
 contactpair[Ilist]=0;
 }
 
 olddispt[Ilist]=olddispti;
 oldfricp[Ilist]=oldfricpi;
 oldLn[Ilist]=j;
 }
}




__global__ void calculateforceiwD(
                       uint  numBodies,
                       double3 worldSize, 
                       double3* pos,
	               double3* pdis,
	               double3* angv,
	               double* rad,
	               double* rmass,
                   double  dt,
                   uint totalcontacts,

                   // return value
                  double3* disptw,
                  double* fricpw,
                  double3* forcei,
                  double3* torquei)
{
//parallerized loop over particle index.
int index = blockIdx.x*blockDim.x + threadIdx.x;  
if(index<numBodies)
{
short ihit;
double3 forceiw,toriw,dptiw,olddptiw;
double fciw,oldfciw;

 olddptiw=disptw[index];
 oldfciw=fricpw[index];

 double3 posi = pos[index];
 double3 pdisi = pdis[index];
 double radi = rad[index];
 double rmassi= rmass[index];
 double3  angvi = angv[index];
 // gravity
 double3 fgrav = { 0.0f, 0.0f, -rmassi };
 double3 force,torque;  

 
 force=make_double3(0.0f);
 torque=make_double3(0.0f);
 if(totalcontacts!=0)
 { 
 force=forcei[index];
 torque=torquei[index];
 }
// add gravity to forcei
 force +=fgrav;

// particle-wall contact force 
 calforceiw(worldSize,posi, pdisi,angvi,radi,rmassi,
            olddptiw,oldfciw,dt,&dptiw,&fciw,&forceiw,&toriw,&ihit);

if(ihit!=0)
{

 force += forceiw;
 torque+=toriw;
 disptw[index]=dptiw;
 fricpw[index]=fciw;
}
else{
 disptw[index]=make_double3(0.0f);
 fricpw[index]=0.0;
}

forcei[index]=force;
torquei[index]=torque;
}
}

__global__ void updatepositionD(
	    double3 worldSize,
        uint  numBodies,
        double3* pos, //input, return
        double3* pdis, //input, return
        double3* angv,   //input, return
        double* rmass,  
        double* inert, 

        double3* forcei,
        double3* torquei,
        double3* qi,    //input, return

		 double  dt,
		 double cdamp,
// return value
         uint* partiall,
         double* partmaxhz)//return
{
    __shared__ double cachemaxhz[256];  
    __shared__ uint cacheiall[256]; 
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int cacheindex=threadIdx.x;

    if(index<numBodies)
    {
    double rmov;
    double ann=0.5*(cub-cua);

// data need to update

   double3 posi = pos[index];
   double3 pdisi = pdis[index];
   double3 angvi= angv[index];
   double3 qqi=qi[index];

// data constant
   double rmassi=rmass[index];
   double inerti=inert[index];

   double3 fi = forcei[index];
   double3 rmomi = torquei[index];

//-------------------------------------------
//--------translational motions -------------
//-------------------------------------------
    double adamp=1.0;
    double bdamp=1.0;
	    pdisi=adamp*pdisi+bdamp*fi*dt*dt/rmassi;
	    posi+=pdisi;
            qqi+= pdisi;
            rmov=length(qqi);
        angvi=adamp*angvi+bdamp*rmomi*dt*dt/inerti;

//--------------------------------------------------
//data after updating
	pos[index]=posi;
	pdis[index]=pdisi;
        qi[index]=qqi;
       angv[index]=angvi;
//calculate maxisum hz 
      cachemaxhz[cacheindex]=posi.z;
       if(rmov>ann) // || posi.z <0.0
        {
         cacheiall[cacheindex]=1;
       }
       else 
         cacheiall[cacheindex]=0;

     if(posi.z<0.0)
     { //printf("after update,lower than the bottom,posi.z=%15.6f,fi.x=%15.6f,%15.6f,%15.6f\n",posi.z,fi.x,fi.y,fi.z); 
      return;}


      } //if(index<numBodies)
      else
      {
       cachemaxhz[cacheindex]=0.0;
       cacheiall[cacheindex]=0;
      }

      __syncthreads();

      int i=blockDim.x/2;
      while(i!=0)
	  {
        if(cacheindex<i)
            {
            cachemaxhz[cacheindex] =fmax(cachemaxhz[cacheindex],cachemaxhz[cacheindex+i]);
	     cacheiall[cacheindex] =max(cacheiall[cacheindex],cacheiall[cacheindex+i]);
			}
        __syncthreads(); 
        i/=2;
      }

     if(cacheindex==0)
     {
     partmaxhz[blockIdx.x]=cachemaxhz[0];
     partiall[blockIdx.x]=cacheiall[0];
     }
//--------------------------------
}


#endif
