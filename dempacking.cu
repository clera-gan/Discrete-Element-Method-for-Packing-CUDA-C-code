/*******************************************************************
mainprog.cu:
This is a part of the DEM code programmed by Zongyan Zhou in fortran.
Modified to C code by Jieqing Gan for the ellipsoidal particles.
Copyright (C) 1998 Particle Technology Group.
The University of New South Wales.
All rights reserved.
********************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
#include "particleSystem.cu"

#define maxCnPerParticle 30
#define m_maxParticlesPerCell 30
//#define uint unsigned int

double dmt1,dmt2,zht1,zht2,dmt15,dmt25,hgt1,hgt2,hopzt;
double dt,pi,diam,thick,thick5,height0;

int main(int argc, char* argv[])
{

   FILE *fp1,*fp6;
// CPU parameters
// particle properties -------------
     double *m_hpos,*m_hpdis,*m_hrad,*m_hangv;
	 double *m_hinert,*m_hrmass,*m_hforcei;
	 uint m_hiall;
	 double m_hmaxhz;

  // declare CPU
  uint *m_holdLn;
  double *m_holddispt,*m_holdfricp,*m_holddisptw,*m_holdfricpw;
 uint *m_holdsjgi;
 uint m_totalcontacts;
 // post process data
uint m_hpcntot;
	struct xpar
	{ 
	 double xp0;
	 double yp0;
	};
	 struct xpar *pxp; // size 200  1
     Matproperties m_hPar;
     Wallproperties m_hParW;

// input data
     int iforcemode,ntot;
     double emod,emodw,denp;      
     double dt,dmt1,dmt2,thick,zht1, zht2;	
     double vel0, nrings,nringsy,cdamp;
     double stiffness, dampcoeffn, dampcoefft;
     uint tstop;
 
//--------------------------
//	 int *icolor,*ncn;
     double *rring,*rringy;      
     double fac,gg;
     double rno,dringsx,dringsy,zp0,dxy,ang;
//-----process data
	 int nlayers,ilay,nparticledat; 
	 long itime,ip;

	 double real_height,real_d2,real_dt,real_thick;
	 double real_height0;
        double realtime,time,tnewpar,tsd,tret,start,finish;
	 double xvolumespace,porosity5,bedaver,avercn;

     char str1[6];
     long i,ir,ij,nplayer,it; 

// output data 
	 int nrestart;

// GPU parameters
uint m_numParticles,layer;
uint m_maxcontacts;

uint m_gridSize[3],m_nGridCells;
double m_cellSize[3],m_worldOrigin[3],m_worldSize[3];

//     declare device array
double *m_dpos,*m_dpdis,*m_drad,*m_dangv,*m_drmass,*m_dinert;

uint *m_dParticleHash[2],*m_dCellStart;
uint *m_dAnei,*m_dnjgi,*m_dnjli,*m_dAref; 
uint *m_dLp,*m_dLn,*m_doldLn,*m_dcontacti,*m_dcontactpair; uint *m_dsjgi,*m_doldsjgi;

double *m_ddispt,*m_dfricp,*m_dolddispt,*m_doldfricp;
double *m_dforcepair,*m_dtorqpairi,*m_dtorqpairj;
double *m_ddisptw,*m_dfricpw;
double *m_dforcei,*m_dtorquei,*m_dqi;

//    ******************************************************************
//    *	         read initial values to parameters	              *
//    ******************************************************************
//		   inputparameters();
//-------------------------
      if((fp1=fopen("hopp3d.inp","rb"))==NULL){
      printf("Cannot open file hopp3d.inp !");
      exit(1);
	   }
//-----define the youngs modulus, major semi-axis, and density
       fscanf(fp1, "%d", &iforcemode);	
       fscanf(fp1, "%lf %lf %lf",&emod,  &m_hPar.vpois, &denp);
       fscanf(fp1, "%lf", &diam);
//-------------------------------------
       fscanf(fp1, "%d", &ntot);
       fscanf(fp1," %lf %u", &dt, &tstop);
       fscanf(fp1," %lf %lf %lf", &dmt1, &dmt2, &thick);
       fscanf(fp1," %lf %lf", &zht1, &zht2);
	
       fscanf(fp1," %lf %d", &vel0, &nrings);
       fscanf(fp1," %lf %lf", &m_hPar.ufricp, &m_hParW.ufricw);
       fscanf(fp1," %lf %lf %lf %lf %lf", &m_hPar.dampnp, &m_hPar.damptp, &m_hParW.dampnw, &m_hParW.damptw,&cdamp);
       fscanf(fp1," %lf %lf %lf", &stiffness, &dampcoeffn, &dampcoefft);
       fscanf(fp1," %lf %lf", &m_hPar.upp, &m_hParW.upw);
       fclose(fp1);
//---------output data for check----------------------------
       printf("%d\n", iforcemode);	
       printf("%lf %lf %lf\n",emod,  m_hPar.vpois, denp);
       printf("%lf\n",diam);      
       printf("%d\n", ntot);
       printf("%lf %u\n",dt, tstop);
       printf("%lf %lf %lf\n", dmt1, dmt2, thick);
       printf("%lf %lf\n", zht1, zht2);	
       printf("%lf %d\n", vel0, nrings);
       printf("%lf %lf\n", m_hPar.ufricp, m_hParW.ufricw);
       printf("%lf %lf %lf %lf %lf\n", m_hPar.dampnp, m_hPar.damptp, m_hParW.dampnw, m_hParW.damptw, cdamp);
       printf("%lf %lf %lf\n", stiffness, dampcoeffn, dampcoefft);
       printf("%lf %lf\n", m_hPar.upp, m_hParW.upw);
//--------------------------------------------------------
        emodw=emod;
        m_hParW.vpoisw=m_hPar.vpois;
        //diama=1.0;
        //diama5=diama*0.5;
//-------------------------------------
        emod=6.0*emod/(3.14159265*denp*9.81*diam);
        emodw=6.0*emodw/(3.14159265*denp*9.81*diam);
        m_hPar.estarp=1.33333333*emod/(2.0*(1.0-m_hPar.vpois*m_hPar.vpois));
        m_hParW.estarw=1.33333333/((1.0-m_hParW.vpoisw*m_hParW.vpoisw)/emod+(1.0-m_hParW.vpoisw*m_hParW.vpoisw)/emodw);
//---------------------
       //real_height=zht2;
       //real_d2=dmt2;
       //real_dt=dt*sqrt(diam/9.81);
       real_thick=thick;
//    ******************************************************************
//    *        set some constants that will be used in the code        *
//   ******************************************************************

//-------------------------
       pi=atan(1.0)*4.0;
       realtime=3600.0;
       //gg=9.81;
       //fac=(pi*denp*gg*diam*diam*diam)/6.0;  
//       adamp=(2.0-alpha*dt)/(2.0+alpha*dt);
//       bdamp=2.0/(2.0+alpha*dt);
       real_height0=0.0;
       height0=real_height0/diam;     
//    ******************************************************************/
//    *       set the maxmium particle diameter 1, and the reduced     */
//    *       units were used                                          */
//    ******************************************************************/
//----- particle property -----------------------------------------------
	   m_hpos=(double *)malloc(ntot*3*sizeof(double));
       if(m_hpos==NULL)exit(1);
	   m_hpdis=(double *)malloc(ntot*3*sizeof(double));
       if(m_hpdis==NULL)exit(1);
	   m_hangv=(double *)malloc(ntot*3*sizeof(double));
       if(m_hangv==NULL)exit(1);
	   m_hforcei=(double *)malloc(ntot*3*sizeof(double));
       if(m_hforcei==NULL)exit(1);
//-------------------------------------------------------------------------
	   m_hrad=(double *)malloc(ntot*sizeof(double));
       if(m_hrad==NULL)exit(1);
	   m_hinert=(double *)malloc(ntot*sizeof(double));
       if(m_hinert==NULL)exit(1);
	   m_hrmass=(double *)malloc(ntot*sizeof(double));
       if(m_hrmass==NULL)exit(1);

	   m_holdsjgi=(uint *)malloc(ntot*sizeof(uint));
       if(m_holdsjgi==NULL)exit(1);
	   m_holdLn=(uint *)malloc(maxCnPerParticle*ntot*sizeof(uint));
       if(m_holdLn==NULL)exit(1);

  	   m_holddispt=(double *)malloc(maxCnPerParticle*ntot*3*sizeof(double));
       if(m_holddispt==NULL)exit(1);
  	   m_holdfricp=(double *)malloc(maxCnPerParticle*ntot*sizeof(double));
       if(m_holdfricp==NULL)exit(1);

	   m_holddisptw=(double *)malloc(ntot*3*sizeof(double));
       if(m_holddisptw==NULL)exit(1);
	   m_holdfricpw=(double *)malloc(ntot*sizeof(double));
       if(m_holdfricpw==NULL)exit(1);

	   memset(m_holdsjgi,0,ntot*sizeof(uint));
	   memset(m_holdLn,0,maxCnPerParticle*ntot*sizeof(uint));

	   memset(m_holddispt,0,maxCnPerParticle*ntot*3*sizeof(double));
	   memset(m_holdfricp,0,maxCnPerParticle*ntot*sizeof(double));
	   memset(m_holddisptw,0,ntot*3*sizeof(double));
	   memset(m_holdfricpw,0,ntot*sizeof(double));
//---------------------
// contain geomery: golbal variables ---
	    thick=real_thick/diam;
        thick5=thick*0.50;	
        dmt1=dmt1/diam;
        dmt2=dmt2/diam;
        zht1=zht1/diam;
        zht2=zht2/diam;
        dmt15=dmt1*0.50;
        dmt25=dmt2*0.50;
        hgt1=zht1;
        hgt2=hgt1+zht2;
        hopzt=hgt2;
        m_hPar.upp=m_hPar.upp/diam;
        m_hParW.upw=m_hParW.upw/diam;
	    
	  if((fp6=fopen("view.dat","wb+"))==NULL){
      printf("Cannot open file view.dat 1#! %d\n");
      exit(0);
	  }
       fprintf(fp6,"number of particles=%4d\n", ntot);
       fprintf(fp6,"diameters =%6.3f\n", diam);
       fprintf(fp6,"time step=%f bed width=%6.3f thickniss=%6.3f\n", dt, dmt1,dmt2);
       fprintf(fp6,"glabal damping=%10.4f bed hbot=%10.4f htop=%10.4f\n", cdamp,hgt1,hgt2);
       printf("number of particles=%4d\n", ntot);
       printf("diameters=%6.3f,%6.3f,%6.3f,ishape= %d\n", diam);
       printf("time step=%fbed width=%6.3f thickniss=%6.3f\n", dt, dmt1,dmt2);
       printf("glabal damping=%10.4f bed hbot=%10.4f htop=%10.4f\n", cdamp,hgt1,hgt2);	   
//   -------------------------------------------------------------------
//   set particle constant properties and initial positions and velocity*
//   -------------------------------------------------------------------
       ip=0;
       nrings=(uint)(dmt2)-3;
       nringsy=(uint)(thick)-3;
	   rring=(double *)malloc(nrings*sizeof(double));
       if(rring==NULL)exit(1);
       rringy=(double *)malloc(nringsy*sizeof(double));
       if(rringy==NULL)exit(1);

	   pxp=(struct xpar *)malloc((nrings*nringsy)*sizeof(struct xpar));
       if(pxp==NULL)exit(1);

       if(dmt25>1.99)
	   {
      dringsx=dmt2/(float)(nrings);

       for(ir=0;ir<nrings;ir++)
	   { 
       rring[ir]=((float)(ir)+0.5)*dringsx-0.5*dmt2;
	   }

       dringsy=thick/(float)(nringsy);
       for(ir=0;ir<nringsy;ir++)
        {
        rringy[ir]=((float)(ir)+0.5)*dringsy-0.5*thick;
        }

       for(ir=0;ir<nrings;ir++)
	 { 
          for(ij=0;ij<nringsy;ij++)
	     {
           pxp[ip].xp0=rring[ir]; // local position
           pxp[ip].yp0=rringy[ij];
           ip=ip+1;
           }
        }
          nplayer=ip;
        } 
        else 
        {
        nplayer=1;
        }

       
       nlayers=(int)(ntot/nplayer)+1;
       free(rring);
       free(rringy);

       memset(m_hangv,0,3*ntot*sizeof(double));

       i=0;
       zp0=hopzt-5.0;
       for(ilay=0;ilay<nlayers;ilay++)
	   {
       for(ip=0;ip<nplayer;ip++)
	   {
         if(i==ntot)break;
           m_hpos[i*3]=pxp[ip].xp0;
           m_hpos[i*3+1]=pxp[ip].yp0;
           m_hpos[i*3+2]=zp0;

           rno= rand()/(double)(RAND_MAX);
           ang=pi*(rno*.33333+.33333);
           m_hpdis[i*3+2]=-vel0*sin(ang)*dt;
           dxy=-vel0*cos(ang)*dt;
           rno= rand()/(double)(RAND_MAX);
           ang=2.0*pi*rno;
           m_hpdis[i*3]=dxy*cos(ang);
           m_hpdis[i*3+1]=dxy*sin(ang);
        m_hrad[i]=0.5;
        m_hrmass[i]=8*m_hrad[i]*m_hrad[i]*m_hrad[i];
        m_hinert[i]=0.4*m_hrmass[i]*(m_hrad[i]*m_hrad[i]);
         i=i+1;
      }
     }
     free(pxp);
     printf("CPU right until now2\n");
//    ******************************************************************
//    *	  initial host golbal variables                       *
//    ******************************************************************
       time=0.0;
       it=0;
	layer=0;
       nparticledat=0;
	m_numParticles=0;
       m_hiall=1;
// ******************************************************************
// initial global variables
    m_cellSize[0]=1.55;
    m_cellSize[1]=1.55;
    m_cellSize[2]=1.55;

    m_worldOrigin[0] = -dmt25;
    m_worldOrigin[1] = -thick5;
    m_worldOrigin[2] = 0.0;

    m_worldSize[0] = dmt2;
    m_worldSize[1] = thick;
    m_worldSize[2] = hopzt;
    m_gridSize[0] = (int)(m_worldSize[0]/m_cellSize[0])+1;
    m_gridSize[1] = (int)(m_worldSize[1]/m_cellSize[1])+1;
    m_gridSize[2] = (int)(m_worldSize[2]/m_cellSize[2])+1;

    m_nGridCells = m_gridSize[0]*m_gridSize[1]*m_gridSize[2];
 
// GPU start 
    // Choose which GPU to run on, change this on a multi-GPU system.
    int   ni=0;
    cudaGetDeviceCount(&ni);
    cudaSetDevice(2);
    cudaRuntimeGetVersion(&ni);
    cudaDriverGetVersion(&ni);

    gpuErrchk(cudaMalloc( (void**)&m_dCellStart, m_nGridCells*sizeof(uint)));
    gpuErrchk(cudaMemcpyToSymbol(Par,&m_hPar,sizeof(Matproperties)));
    gpuErrchk(cudaMemcpyToSymbol(ParW,&m_hParW,sizeof(Wallproperties)));
//**********************************************************************
//    *    if the value of nrestart is zero, { read data from the   *
//    *     file of preflow.dat                                        *
//**********************************************************************
 /*   if((fp1=fopen("restart.dat","rb+"))==NULL){
    printf("Cannot open file restart.dat!");
    exit(0);
	}
    fscanf(fp1,"%d %d %d",&nrestart,&npregas,&nparticledat);
	fclose(fp1);
*/
//    ******************************************************************
//    ******************************************************************
//    *             main program starts here			      *
//    ******************************************************************
    start= clock();
	while( it< tstop) 
 	{    
     it=it+1;
     time=dt*it;
	   // poured packing
     if(m_numParticles<ntot)
	 {
     tnewpar=0.50;
     if(time>(float)m_numParticles*tnewpar/float(nplayer))
	 {
      // free memery with old size
	  layer++;
//	  printf("CPU right until now3\n");
	  if(layer>1)
	 {		
	  m_maxcontacts=m_numParticles*maxCnPerParticle;
/*  for(i=0;i<m_numParticles;i++)
	  {printf("particle %d position x=%10.7f,y=%10.7f,z=%10.7f\n",i, m_hpos[i*3],m_hpos[i*3+1],m_hpos[i*3+2]);}*/
  gpuErrchk(cudaMemcpy(m_hpos, m_dpos,m_numParticles*3*sizeof(double),
	                        cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(m_hpdis, m_dpdis,m_numParticles*3*sizeof(double),
	                        cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(m_hangv, m_dangv,m_numParticles*3*sizeof(double),
	                        cudaMemcpyDeviceToHost));

  gpuErrchk(cudaMemcpy(m_hrad, m_drad,m_numParticles*sizeof(double),
	                        cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(m_hrmass, m_drmass,m_numParticles*sizeof(double),
	                        cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(m_hinert, m_dinert,m_numParticles*sizeof(double),
	                        cudaMemcpyDeviceToHost));

  gpuErrchk(cudaMemcpy(m_holdsjgi,m_dsjgi,m_numParticles*sizeof(uint),
	                        cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(m_holdLn, m_doldLn,m_maxcontacts*sizeof(uint),
	                        cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(m_holddispt, m_dolddispt,m_maxcontacts*3*sizeof(double),
	                        cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(m_holdfricp, m_doldfricp,m_maxcontacts*sizeof(double),
	                        cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(m_holddisptw, m_ddisptw,m_numParticles*3*sizeof(double),
	                        cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(m_holdfricpw, m_dfricpw,m_numParticles*sizeof(double),
	                        cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(m_dpos));
	gpuErrchk(cudaFree(m_dpdis));
	gpuErrchk(cudaFree(m_dangv));

	gpuErrchk(cudaFree(m_drad));
	gpuErrchk(cudaFree(m_drmass));
	gpuErrchk(cudaFree(m_dinert));

       gpuErrchk(cudaFree(m_dParticleHash[0]));
	gpuErrchk(cudaFree(m_dParticleHash[1]));

	gpuErrchk(cudaFree(m_dAnei));
	gpuErrchk(cudaFree(m_dAref));
	gpuErrchk(cudaFree(m_dnjgi));
	gpuErrchk(cudaFree(m_dnjli));
	gpuErrchk(cudaFree(m_dsjgi));
	gpuErrchk(cudaFree(m_doldsjgi));

	gpuErrchk(cudaFree(m_dLp));
	gpuErrchk(cudaFree(m_dLn));
	gpuErrchk(cudaFree(m_doldLn));

	gpuErrchk(cudaFree(m_ddispt));
	gpuErrchk(cudaFree(m_dfricp));
	gpuErrchk(cudaFree(m_dolddispt));
	gpuErrchk(cudaFree(m_doldfricp));

	gpuErrchk(cudaFree(m_dforcepair));
	gpuErrchk(cudaFree(m_dtorqpairi));
	gpuErrchk(cudaFree(m_dtorqpairj));
	gpuErrchk(cudaFree(m_dcontactpair));

   gpuErrchk(cudaFree(m_ddisptw));
   gpuErrchk(cudaFree(m_dfricpw));

   gpuErrchk(cudaFree(m_dforcei));
   gpuErrchk(cudaFree(m_dtorquei));
   gpuErrchk(cudaFree(m_dcontacti));
   gpuErrchk(cudaFree(m_dqi));
		}

       m_numParticles=m_numParticles+nplayer;
       if(m_numParticles>ntot)m_numParticles=ntot;
       m_hiall=1;
	   m_maxcontacts=m_numParticles*maxCnPerParticle;

// allocate memery for new array with new size 
 
   gpuErrchk(cudaMalloc((void**)&m_dpos, m_numParticles*3*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_dpdis, m_numParticles*3*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_dangv, m_numParticles*3*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_drad, m_numParticles*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_drmass, m_numParticles*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_dinert, m_numParticles*sizeof(double)));

/*  for(i=0;i<m_numParticles;i++)
	  {
	printf("particle %d position x=%10.7f,y=%10.7f,z=%10.7f\n",i, m_hpos[i*3],m_hpos[i*3+1],m_hpos[i*3+2]);}*/
   gpuErrchk(cudaMemcpy(m_dpos, m_hpos,m_numParticles*3*sizeof(double),
	                        cudaMemcpyHostToDevice));
   gpuErrchk(cudaMemcpy(m_dpdis, m_hpdis,m_numParticles*3*sizeof(double),
	                        cudaMemcpyHostToDevice));
   gpuErrchk(cudaMemcpy(m_dangv, m_hangv,m_numParticles*3*sizeof(double),
	                        cudaMemcpyHostToDevice));
   gpuErrchk(cudaMemcpy(m_drad, m_hrad,m_numParticles*sizeof(double),
	                        cudaMemcpyHostToDevice));
   gpuErrchk(cudaMemcpy(m_drmass, m_hrmass,m_numParticles*sizeof(double),
	                        cudaMemcpyHostToDevice));
   gpuErrchk(cudaMemcpy(m_dinert, m_hinert,m_numParticles*sizeof(double),
	                        cudaMemcpyHostToDevice));

   gpuErrchk(cudaMalloc((void**)&m_dParticleHash[0], m_numParticles*2*sizeof(uint)));
   gpuErrchk(cudaMalloc((void**)&m_dParticleHash[1], m_numParticles*2*sizeof(uint)));

   gpuErrchk(cudaMalloc((void**)&m_dAnei, m_numParticles*m_maxParticlesPerCell*sizeof(uint)));
   gpuErrchk(cudaMalloc((void**)&m_dAref, m_numParticles*m_maxParticlesPerCell*sizeof(uint)));
   gpuErrchk(cudaMalloc((void**)&m_dnjgi, m_numParticles*sizeof(uint)));
   gpuErrchk(cudaMalloc((void**)&m_dnjli, m_numParticles*sizeof(uint)));
   gpuErrchk(cudaMalloc((void**)&m_dsjgi, m_numParticles*sizeof(uint)));
   gpuErrchk(cudaMalloc((void**)&m_doldsjgi, m_numParticles*sizeof(uint)));

   gpuErrchk(cudaMalloc((void**)&m_dLp, m_maxcontacts*sizeof(uint)));
   gpuErrchk(cudaMalloc((void**)&m_dLn, m_maxcontacts*sizeof(uint)));
   gpuErrchk(cudaMalloc((void**)&m_doldLn, m_maxcontacts*sizeof(uint)));

   gpuErrchk(cudaMalloc((void**)&m_ddispt, m_maxcontacts*3*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_dfricp, m_maxcontacts*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_dolddispt, m_maxcontacts*3*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_doldfricp, m_maxcontacts*sizeof(double)));

   gpuErrchk(cudaMalloc((void**)&m_dforcepair, m_maxcontacts*3*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_dtorqpairi, m_maxcontacts*3*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_dtorqpairj, m_maxcontacts*3*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_dcontactpair, m_maxcontacts*sizeof(uint)));

   gpuErrchk(cudaMalloc((void**)&m_ddisptw, m_numParticles*3*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_dfricpw, m_numParticles*sizeof(double)));

   gpuErrchk(cudaMalloc((void**)&m_dforcei, m_numParticles*3*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_dtorquei, m_numParticles*3*sizeof(double)));
   gpuErrchk(cudaMalloc((void**)&m_dcontacti, m_numParticles*sizeof(uint)));
   gpuErrchk(cudaMalloc((void**)&m_dqi, m_numParticles*3*sizeof(double)));
 
    gpuErrchk(cudaMemcpy(m_dsjgi,m_holdsjgi,m_numParticles*sizeof(uint),
	                        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(m_doldLn, m_holdLn,m_maxcontacts*sizeof(uint),
	                        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(m_dolddispt, m_holddispt,m_maxcontacts*3*sizeof(double),
	                        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(m_doldfricp, m_holdfricp,m_maxcontacts*sizeof(double),
	                        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(m_ddisptw, m_holddisptw,m_numParticles*3*sizeof(double),
	                        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(m_dfricpw, m_holdfricpw,m_numParticles*sizeof(double),
	                        cudaMemcpyHostToDevice));
    }
   }
   gpuErrchk(cudaMemset(m_dforcei,0, m_numParticles*3*sizeof(double)));
   gpuErrchk(cudaMemset(m_dtorquei,0, m_numParticles*3*sizeof(double)));
   gpuErrchk(cudaMemset(m_dcontacti,0, m_numParticles*sizeof(uint)));

   if(m_hiall>0) 
   {
   gpuErrchk(cudaMemset(m_dqi,0,m_numParticles*3*sizeof(double)));

// GPU code part: generate neighbour list ---------------
    // calculate hash
    calcHash(m_dpos,
             m_dParticleHash[0],
             m_gridSize,
             m_cellSize,
             m_worldOrigin,
             m_numParticles);

	 // sort particles based on hash
    RadixSort((KeyValuePair *) m_dParticleHash[0], 
		      (KeyValuePair *) m_dParticleHash[1], 
			   m_numParticles, 32);
	 
    // find start of each cell
    findCellStart(m_dParticleHash[0], m_dCellStart, m_numParticles, m_nGridCells);

	// generate neighbour list Anei and number of particles 
	// greater or lesser than particle i
    neighborarray(m_dpos,
		    m_dParticleHash[0],
                  m_dCellStart,
                  m_gridSize,
                  m_cellSize,
                  m_worldOrigin,
                  m_maxParticlesPerCell,
		  m_numParticles,
                  // return value
                  m_dAnei,
                  m_dnjgi,
                  m_dnjli);

	//obtain Sjgi for each partice i
	prefixsum(m_numParticles,
              m_dnjgi,
              // return value
              m_dsjgi,
              m_doldsjgi,
              &m_totalcontacts);
/*

     if(layer>=1){

      uint *m_hAref;
      m_hAref=(uint *)malloc(m_numParticles*m_maxParticlesPerCell*sizeof(uint));

     gpuErrchk(cudaMemcpy(m_hAref,m_dAnei,m_numParticles*m_maxParticlesPerCell*sizeof(uint),cudaMemcpyDeviceToHost));
     for(i=0;i<m_numParticles;i++){
      for(uint k=0; k<m_maxParticlesPerCell;k++){
       printf("m_hAei[%d][%d]=%d ",i,k,m_hAref[i*m_maxParticlesPerCell+k]);}
       printf("\n");
       }
      free(m_hAref);


     gpuErrchk(cudaMemcpy(m_holdLn,m_doldLn,m_maxcontacts*sizeof(uint),cudaMemcpyDeviceToHost));

     for(i=0;i<m_totalcontacts;i++){
       printf("m_holdLn[%d]=%d\n",i,m_holdLn[i]);}


     printf("m_totalcontacts=%d\n",m_totalcontacts);
     gpuErrchk(cudaMemcpy(m_holdsjgi,m_doldsjgi,m_numParticles*sizeof(uint),cudaMemcpyDeviceToHost));

     for(i=0;i<m_numParticles;i++){
       printf("oldsjgi[%d]=%d\n",i,m_holdsjgi[i]);}

     gpuErrchk(cudaMemcpy(m_holdsjgi,m_dsjgi,m_numParticles*sizeof(uint),cudaMemcpyDeviceToHost));

     for(i=0;i<m_numParticles;i++){
       printf("sjgi[%d]=%d\n",i,m_holdsjgi[i]);}

     gpuErrchk(cudaMemcpy(m_holdsjgi,m_dnjgi,m_numParticles*sizeof(uint),cudaMemcpyDeviceToHost));

     for(i=0;i<m_numParticles;i++){
       printf("m_dnjgi[%d]=%d\n",i,m_holdsjgi[i]);}
      exit(1);  
     

     gpuErrchk(cudaMemcpy(m_holdsjgi,m_dnjli,m_numParticles*sizeof(uint),cudaMemcpyDeviceToHost));

     for(i=0;i<m_numParticles;i++){
       printf("m_dnjli[%d]=%d\n",i,m_holdsjgi[i]);}

     gpuErrchk(cudaMemcpy(m_holddispt, m_dolddispt,m_maxcontacts*3*sizeof(double),
	                        cudaMemcpyDeviceToHost));
     gpuErrchk(cudaMemcpy(m_holdfricp, m_doldfricp,m_maxcontacts*sizeof(double),
	                        cudaMemcpyDeviceToHost));
     for(i=0;i<m_numParticles;i++){
       printf("m_holddispt[%d]=%10.6f,m_holddispt[%d]=%10.6f,m_holddispt[%d]=%10.6f,m_holdfricp[%d]=%10.6f\n",i,m_holddispt[3*i],i,m_holddispt[3*i+1],i,m_holddispt[3*i+2],i,m_holdfricp[i]);}
       exit(1);    
  

     }

*/
//--------------------------------------------------------------------------------
// generate contact candidate pair list Lp[Ilist]=i,Ln[Ilist]=j 
// and the reference table Aref
// check whether the current pair were in contact in previous time step, 
// if yes, initial dispt and fricp value for pair list with previous vulue, 
// otherwise initial zero, 
//--------------------------------------------------------------------------------
         pairlist(m_dsjgi,
	      m_doldsjgi,
             m_doldLn,
             m_dAnei,
             m_dnjgi,
             m_dnjli,
	      m_dolddispt,
             m_doldfricp,

             m_maxParticlesPerCell,
             maxCnPerParticle,
             m_numParticles,

             // return value

             m_dAref,
             m_dLp,
             m_dLn,
	      m_ddispt,
             m_dfricp);

    // updated neighbour list
     m_hiall=0;
    }

 // calculate contact force and torque for each candidate pair
	calculateforcepair(m_dpos,m_dpdis,m_dangv,m_drad,m_drmass,
					   m_dLp,
					   m_dLn,
                                      m_doldLn,
		               m_ddispt,
                       m_dfricp,
                       m_dolddispt,
                       m_doldfricp,
                       dt,
                       // return value
                       m_dforcepair,
                       m_dtorqpairi,
					   m_dtorqpairj,
					   m_dcontactpair,
				   m_totalcontacts);


 // sum all the forces on particle i
    sumallforcei( m_maxParticlesPerCell,
                  m_numParticles,
                  m_dnjgi,
                  m_dnjli,

                  m_dforcepair,
                  m_dtorqpairi,
		  m_dtorqpairj,
		  m_dcontactpair,
                  m_dAref,
		  // return
                  m_dforcei,
                  m_dtorquei,
		  m_dcontacti);


    calculateforceiw(m_worldSize,
		    m_dpos,m_dpdis,m_dangv,m_drad,m_drmass,
                  dt,
	           m_totalcontacts,
		    m_numParticles,

                  // return value
                  m_ddisptw,
                  m_dfricpw,

                  m_dforcei,
                  m_dtorquei);
  

	// update particle position, displacement,orientation,
	// angv, and angvb, q0, etc.
	updateposition(m_worldSize,
		       m_numParticles,
		       m_dpos,
                       m_dpdis,
                       m_dangv, 
   
                       m_drmass,
		       m_dinert,

                       m_dforcei,
                       m_dtorquei,
                       m_dqi,
		       dt,
		       cdamp,

                   // return value
		       &m_hiall,
                     &m_hmaxhz);

//printf("it=%d,m_numParticles=%d, m_totalcontacts=%d,m_hiall=%d\n",it,m_numParticles, m_totalcontacts,m_hiall);


//**********************************************************************
// copy position, displacement and orientation data 
// from device to host for checking and output
	if((it%50000)==0)
	{
  gpuErrchk(cudaMemcpy(m_hpos,m_dpos, m_numParticles*3*sizeof(double),cudaMemcpyDeviceToHost));
 // gpuErrchk(cudaMemcpy(m_hpdis,m_dpdis, m_numParticles*3*sizeof(double),
//	                        cudaMemcpyDeviceToHost));
 // gpuErrchk(cudaMemcpy(m_hcontacti,m_dcontacti, m_numParticles*sizeof(uint),
//	                        cudaMemcpyDeviceToHost));
    }
//**********************************************************************
//         store data of particle positons at different time           *
//**********************************************************************
	if((it%1000)==0 )
	{
     outputporositycn(m_numParticles,
                      m_dcontacti,
		              m_hmaxhz,
		              m_worldSize,
                      // return value
                      &m_hpcntot);

	//-------record the density and cn
    avercn=m_hpcntot/(m_numParticles*1.0+1.0e-30);
	xvolumespace=m_worldSize[0]*m_worldSize[1]*(m_hmaxhz);
    porosity5=1.0-m_numParticles*pi/6/xvolumespace;
    bedaver=1.0-porosity5;
    if((fp1=fopen("ppor.dat","ab+"))==NULL)printf("Cannot open file ppor.dat!");
    //tsd=(it)*real_dt;
    fprintf(fp1,"%8d %8d %8d %15.6f %15.6f %15.6f %15.6f\n",it,m_numParticles,m_hpcntot/2,porosity5,bedaver,avercn,m_hmaxhz);
    printf(     "%8d %8d %8d %15.6f %15.6f %15.6f %15.6f\n",it,m_numParticles,m_hpcntot/2,porosity5,bedaver,avercn,m_hmaxhz);
    fclose(fp1);
	}
//**********************************************************************
       if((it%50000)==0 || it==1)
	{
      if((fp1=fopen("particle.dat","ab+"))==NULL){
      printf("Cannot open file particle01.dat! %d\n",it);
      exit(0);}

      nparticledat=nparticledat+1;
      sprintf(str1,"%03d",nparticledat);
      fprintf(fp1,"ZONE T=\"ZONE%s\"\n",str1);

      for(i=0;i<m_numParticles;i++)
      fprintf(fp1,"%15.6g%15.6g%15.6g%15.6g%8d\n",m_hpos[3*i]*diam,m_hpos[3*i+1]*diam,m_hpos[3*i+2]*diam,m_hrad[i]*diam,i);
      fclose(fp1);
      }
//******************************************************************
//    *      write preflow data for re-calculation        *
//******************************************************************
      finish = clock();
      tret=(double)(finish-start)/CLOCKS_PER_SEC;

      if((it%1000)==0)
	  {
      if((fp1=fopen("timesteps.dat","ab+"))==NULL)printf("Cannot open file timesteps.dat!");
      fprintf(fp1,"%15.6f %15.6f\n", tret,it*1.0);
      fclose(fp1);
      }
      if(tret>=realtime)
	  {
       itime=0;
      cudaMemcpy(m_hpos,m_dpos, m_numParticles*3*sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(m_hpdis,m_dpdis, m_numParticles*3*sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(m_hangv,m_dangv, m_numParticles*3*sizeof(double),cudaMemcpyDeviceToHost);

        if((fp1=fopen("preflow.dat","wb+"))==NULL)printf("Cannot open file preflow.dat!");
        fprintf(fp1,"%8d,%8d,%8d\n", it,m_numParticles,itime);

        for(i=0;i<m_numParticles;i++){
        fprintf(fp1,"%15.6f %15.6f %15.6f %15.6f %15.6f %15.6f %15.6f %15.6f %15.6f\n",
			m_hpos[3*i],  m_hpos[3*i+1],  m_hpos[3*i+2],
			m_hpdis[3*i], m_hpdis[3*i+1], m_hpdis[3*i+2],
                     m_hangv[3*i], m_hangv[3*i+1], m_hangv[3*i+2]);
		}

        fclose(fp1);
        realtime=realtime+7200.0;

        if((fp1=fopen("restart.dat","rb+"))==NULL){
        printf("Cannot open file restart.dat!");
        exit(0);
		}
//        fprintf(fp1,"%8d %8d %8d",nrestart,npregas,nparticledat);
        fclose(fp1);
      }
 
//   ******************************************************************
//    *      stop programme temporarily if running time is 	      *
//    *       beyond 7 hours                                           *             
//   ******************************************************************
       if(tret>41400.)
	   {  
        fclose(fp6);
//-----release GPU memory -------------------------	
		cudaFree(m_dpos);
		cudaFree(m_dpdis);
		cudaFree(m_drad);
		cudaFree(m_dangv);
		cudaFree(m_drmass);
		cudaFree(m_dinert);

		cudaFree(m_dParticleHash[0]);
		cudaFree(m_dParticleHash[1]);

		cudaFree(m_dAnei);
		cudaFree(m_dAref);
		cudaFree(m_dnjgi);
		cudaFree(m_dnjli);
		cudaFree(m_dsjgi);
		cudaFree(m_doldsjgi);

		cudaFree(m_dLp);
		cudaFree(m_dLn);
		cudaFree(m_doldLn);

		cudaFree(m_ddispt);
		cudaFree(m_dfricp);
		cudaFree(m_dolddispt);
		cudaFree(m_doldfricp);

		cudaFree(m_dforcepair);
		cudaFree(m_dtorqpairi);
		cudaFree(m_dtorqpairj);
		cudaFree(m_dcontactpair);

		cudaFree(m_ddisptw);
		cudaFree(m_dfricpw);

		cudaFree(m_dforcei);
	    cudaFree(m_dtorquei);
	    cudaFree(m_dcontacti);
             cudaFree(m_dqi);

	    cudaFree(m_dCellStart);
//-------release CPU memory -------------------------	
	 free(m_hpos);
	 free(m_hpdis);
        free(m_hangv);
	 free(m_hrad);
        free(m_hinert);
	 free(m_hrmass);
	 free(m_hforcei);

	 free(m_holdsjgi);
	 free(m_holdLn);
        free(m_holddispt);
        free(m_holdfricp);
        free(m_holddisptw);
        free(m_holdfricpw);

        exit(1);
       } 

} // while
//    ******************************************************************
//    *      finish off the run now                                    *
//    ******************************************************************
       if((fp1=fopen("jobfinished","wb"))==NULL)
		   printf("Cannot open file jobfinished!");
       fprintf(fp1,"job finished\n");
       fclose(fp1);
       if((fp1=fopen("restart.dat","wb"))==NULL)
		   printf("Cannot open file restart.dat");
       fprintf(fp1,"program comes to end!\n");
	   fprintf(fp1,"%8d %8d\n",1,1);
       printf("program comes to end!\n");
       fclose(fp1);
       exit(1);
//	   fclose(fp6);

	return 0;
}

//*************************************************************************
//*************************************************************************
