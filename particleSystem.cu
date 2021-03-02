
#include <cstdlib>
#include <cstdio>
#include <string.h>

#include "neiglist.cu"
#include "radixsort.cu"
#include "scan.cu"
#include "scanLargeArray_kernel.cu"

//int min(int x,int y){return(x<y?x:y);}

extern "C"
{
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



void calcHash(double*  Pos, 
         uint*   particleHash,
         uint    gridSize[3],
         double   cellSize[3],
         double   worldOrigin[3],
         int     numBodies )
{
    int numThreads = min(256, numBodies);
    int numBlocks = (int) ceil(numBodies / (double) numThreads);

    // execute the kernel
    calcHashD<<< numBlocks, numThreads >>>(numBodies,
                                          (double3 *) Pos,
                                           (uint2 *) particleHash,
                                           make_uint3(gridSize[0], gridSize[1], gridSize[2]),
                                           make_double3(cellSize[0], cellSize[1], cellSize[2]),
                                           make_double3(worldOrigin[0], worldOrigin[1], worldOrigin[2]));
    
}



void findCellStart(uint* particleHash,
              uint* cellStart,
              int numBodies,
              uint numGridCells)
{
    // scatter method
    int numThreads = min(numBodies,256);
    uint numBlocks = (uint) ceil(numBodies / (double) numThreads);

    gpuErrchk(cudaMemset(cellStart, 0xffffffff, numGridCells*sizeof(uint)));

    findCellStartD<<< numBlocks, numThreads >>>((uint2 *) particleHash,
                                                cellStart,
                                                numBodies,
                                                numGridCells);
}


void neighborarray(double* pos,
                   uint*	particleHash,
                   uint*	cellStart,
                   uint		gridSize[3],
                   double  cellSize[3],
                   double  worldOrigin[3],
                   int		maxParticlesPerCell,
		           int		numBodies,
                   // return value
                   uint*	Anei,
                   uint*	njgi,
                   uint*	njli)
{
    int numThreads = min(numBodies,256);
    uint numBlocks = (uint) ceil(numBodies / (double) numThreads);

 gpuErrchk(cudaMemset(Anei, 0xffffffff, numBodies*maxParticlesPerCell*sizeof(uint)));

 neighborarrayD<<< numBlocks, numThreads >>>(numBodies,
                                          (double3 *) pos,
                                         (uint2*) particleHash,
              				 cellStart,
	                                 make_uint3(gridSize[0], gridSize[1], gridSize[2]),
	         			 make_double3(cellSize[0], cellSize[1], cellSize[2]),
                                         make_double3(worldOrigin[0], worldOrigin[1], worldOrigin[2]),
                                         maxParticlesPerCell,
                                             // return value
                                         Anei,
                                         njgi,
                                         njli );
}

/*
void prefixsum(uint  numBodies,
               uint* njgi,
               // return value
               uint* sjgi, 
               uint* oldsjgi,
               uint *totalcontacts)
{
 int numThreads = min(numBodies,256);
 uint numBlocks = (uint) ceil(numBodies / (double) numThreads);
 uint *contacts;

 gpuErrchk(cudaMalloc((void**)&contacts,sizeof(uint)));
 gpuErrchk(cudaMemset(contacts, 0,sizeof(uint)));

 prefixsumD<<< numBlocks, numThreads >>>(numBodies,
                                        njgi,
                                        sjgi,
                                        oldsjgi,
					     contacts); 
  gpuErrchk(cudaMemcpy(totalcontacts,contacts,sizeof(uint),cudaMemcpyDeviceToHost));

 cudaFree(contacts);

}
*/

void prefixsum(uint numBodies, uint *njgi, uint *sjgi,uint* oldsjgi,uint *totalcontacts)
{
 int numThreads = min(numBodies,256);
 uint numBlocks = (uint) ceil(numBodies / (double) numThreads);

    gpuErrchk(cudaMemcpy(oldsjgi,sjgi,numBodies*sizeof(uint),cudaMemcpyDeviceToDevice));

    preallocBlockSums(numBodies);
    prescanArray(sjgi, njgi, numBodies);
    addTwoArraysD<<< numBlocks, numThreads >>>(sjgi,njgi,numBodies);

    deallocBlockSums();  
    gpuErrchk(cudaMemcpy(totalcontacts,sjgi+numBodies-1,sizeof(uint),cudaMemcpyDeviceToHost));

}


void pairlist(uint* sjgi,
              uint* oldsjgi,
              uint* oldLn,
              uint* Anei,
              uint* njgi,
              uint* njli,
              double* olddispt,
              double* oldfricp,

              int   maxParticlesPerCell,
              int   maxCnPerParticle,
              uint  numBodies,

              // return value
              uint* Aref,
              uint* Lp,
              uint* Ln,
              double* dispt,
              double* fricp)
{
 int numThreads = min(numBodies,256);
 uint numBlocks = (uint) ceil(numBodies/ (double) numThreads);

 gpuErrchk(cudaMemset(Aref, 0xffffffff, numBodies*maxParticlesPerCell*sizeof(uint)));
 gpuErrchk(cudaMemset(Lp, 0xffffffff, numBodies*maxCnPerParticle*sizeof(uint)));
 gpuErrchk(cudaMemset(Ln, 0xffffffff, numBodies*maxCnPerParticle*sizeof(uint)));
 gpuErrchk(cudaMemset(dispt, 0, numBodies*maxCnPerParticle*3*sizeof(uint)));
 gpuErrchk(cudaMemset(fricp, 0, numBodies*maxCnPerParticle*sizeof(uint)));

 pairlistD<<< numBlocks, numThreads >>>( sjgi,
                                         oldsjgi,
                                         oldLn,
                                         Anei,
                                         njgi,
                                         njli,
					      (double3 *) olddispt,
                                         oldfricp,

                                         maxParticlesPerCell,
					      numBodies,

                                         // return value                                     

                                         Aref,
                                         Lp,
                                         Ln,
                                         (double3 *) dispt,
                                         fricp);
}


void calculateforcepair(double* pos,
	               double* pdis,
	               double* angv,
	               double* rad,
	               double* rmass,
		     uint* Lp,
                   uint* Ln,
                   uint* oldLn,

                   double*  dispt,
                   double*  fricp,
                   double*  olddispt,
                   double* oldfricp,
                   double  dt,

                   // return value
                   double* forcepair,
                   double* torqpairi,
				   double* torqpairj,
				   uint* contactpair,
				   uint	totalcontacts)
{
 int numThreads = min(totalcontacts,256);
 uint numBlocks = (uint) ceil(totalcontacts/ (double) numThreads);


 calculateforcepairD<<< numBlocks, numThreads >>>(totalcontacts,
                                                      (double3 *) pos,
	                                              (double3 *) pdis,
	                                              (double3 *) angv,
	                                               rad,
	                                              rmass,
	                                              Lp,
	                                              Ln,
                                                     oldLn,
	                                              (double3 *) dispt,
                                                  fricp,
                                                  (double3 *) olddispt,
                                                  oldfricp,
                                                  dt,
                                                  // return value
						          (double3 *) forcepair,
		           			          (double3 *) torqpairi,
						          (double3 *) torqpairj,
						           contactpair);	   

}


void sumallforcei(int   maxParticlesPerCell,
	           int	numBodies,
                  uint* njgi,
                  uint* njli,

                  double* forcepair,
                  double* torqpairi,
		    double* torqpairj,
		    uint* contactpair,

                  uint* Aref,
		    //return
                  double* forcei,
                  double* torquei,
		    uint* contacti)
{
 int numThreads = min(numBodies,256);
 uint numBlocks = (uint) ceil(numBodies/ (double) numThreads);

 sumallforceiD<<< numBlocks, numThreads >>>(numBodies,
                                            maxParticlesPerCell,
                                            njgi,
               			            njli,

                                            (double3 *) forcepair ,
					         (double3 *) torqpairi,
					         (double3 *) torqpairj,
					          contactpair,
                                             Aref,

                                            // return value 
                                            (double3 *) forcei,
                                            (double3 *) torquei,
					         contacti);
}

void calculateforceiw(double worldSize[3], 
                   double* pos,
	               double* pdis,
	               double* angv,
	               double* rad,
	               double* rmass,
                   double  dt,
                   uint    totalcontacts,
                   int	numBodies,

                   // return value 
                   double* disptw,
                   double* fricpw,
                   double* forcei,
                   double* torquei)
{
 int numThreads = min(numBodies,256);
 uint numBlocks = (uint) ceil(numBodies/ (double) numThreads);
 calculateforceiwD<<< numBlocks, numThreads >>>(numBodies, 
                              make_double3(worldSize[0], worldSize[1], worldSize[2]),
                              (double3 *) pos,
	                      (double3 *) pdis,
	                      (double3 *) angv,
	                       rad,
	                       rmass,
                              dt,
                              totalcontacts,

                              // return value
                              (double3 *) disptw,
                               fricpw,
                              (double3 *) forcei,
                              (double3 *) torquei);
 //gpuErrchk(cudaGetLastError());
}

void updateposition(double worldSize[3],
		int	numBodies,

        double* pos,  //old,new
        double* pdis, //old,new
        double* angv,   //old,new
        double* rmass,  
        double* inert, 

        double* forcei,
        double* torquei,
        double* qi,

		 double  dt,
		 double cdamp,
                 // return value
		 uint *iall,
               double* maxhz)
{

  int numThreads = 256;
 uint numBlocks = (uint) ceil(numBodies/ (double) numThreads);
 uint *dpartialiall,*hpartialiall;
 double *dpartialmaxhz,*hpartialmaxhz;

 hpartialiall=(uint *)malloc(numBlocks*sizeof(uint));
 if(hpartialiall==NULL)exit(1);
 memset(hpartialiall,0, numBlocks*sizeof(uint));

 hpartialmaxhz=(double *)malloc(numBlocks*sizeof(double));
 if(hpartialmaxhz==NULL)exit(1);
 memset(hpartialmaxhz,0, numBlocks*sizeof(double));

 gpuErrchk(cudaMalloc( (void**)&dpartialmaxhz, numBlocks*sizeof(double)));
 gpuErrchk(cudaMemset(dpartialmaxhz, 0,  numBlocks*sizeof(double)));
 gpuErrchk(cudaMalloc( (void**)&dpartialiall, numBlocks*sizeof(uint)));
 gpuErrchk(cudaMemset(dpartialiall, 0,  numBlocks*sizeof(uint)));

 updatepositionD<<< numBlocks, numThreads >>>( make_double3(worldSize[0], worldSize[1], worldSize[2]),
	                 		       numBodies,
	                                       (double3 *) pos,  //old,new
                                              (double3 *) pdis, //old,new
                                              (double3 *) angv,   //old,new
					            rmass,
                                               inert, //old,new
 		                                (double3 *) forcei,
                                              (double3 *) torquei,   
                                              (double3 *) qi,

                                               dt,cdamp,
                                              // return value
                                               dpartialiall,
						     dpartialmaxhz);

gpuErrchk(cudaMemcpy(hpartialiall,dpartialiall,numBlocks*sizeof(uint),cudaMemcpyDeviceToHost));
gpuErrchk(cudaMemcpy(hpartialmaxhz,dpartialmaxhz,numBlocks*sizeof(double),cudaMemcpyDeviceToHost));

 *iall=0; 
  *maxhz=0.0;

 for(uint i=0;i<numBlocks;i++)
 {
*iall= max(*iall,hpartialiall[i]);
*maxhz= fmax(*maxhz,hpartialmaxhz[i]);
 }
 free(hpartialiall);
 free(hpartialmaxhz);
 cudaFree(dpartialiall);
 cudaFree(dpartialmaxhz);
}

void outputporositycn(uint numBodies,   
                     uint* contacti,
		             double maxhz,
		             double worldSize[3],
// return value
                     uint* pcntot)
{

 int numThreads = 256;
 uint numBlocks = (uint) ceil(numBodies/ (double) numThreads);
 uint *dpartpcntot,*hpartpcntot;

  hpartpcntot=(uint *)malloc(numBlocks*sizeof(uint));
 if(hpartpcntot==NULL)exit(1);
 memset(hpartpcntot,0, numBlocks*sizeof(uint));

 gpuErrchk(cudaMalloc( (void**)&dpartpcntot, numBlocks*sizeof(uint)));
 gpuErrchk(cudaMemset(dpartpcntot, 0,  numBlocks*sizeof(uint)));

 outputporositycnD<<< numBlocks, numThreads >>>(numBodies,
					        contacti,
						 maxhz,
						 make_double3(worldSize[0], worldSize[1], worldSize[2]),
						 // return value
						 dpartpcntot);

 gpuErrchk(cudaMemcpy(hpartpcntot,dpartpcntot,numBlocks*sizeof(uint),cudaMemcpyDeviceToHost));

 *pcntot=0;
 for(uint i=0;i<numBlocks;i++)
 {
 *pcntot +=hpartpcntot[i];
 }

 free(hpartpcntot);
 cudaFree(dpartpcntot);
}

}   // extern "C"
