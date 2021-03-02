# Discrete-Element-Method-for-Packing-CUDA-C-code
CUDA C version of particle packing to realize the same functions as the FORTRAN and C++ codes in the other two repos.</br>

This code is based on the algorithm parallelized by contact candidate pair for force calculation (see Figure 1 for the difference of two parallelization algorithms for force calculation).</br>

Figure. 1 (a) Flowchart of DEM simulation by GPU, (b) algorithm parallelized by particles, and (c) algorithm parallelized by contact candidate pair for force calculation. </br>

Figure. 2 Performance of DEM by CPU and a single GPU with spherical particles: (a) elapsed time per second with particle number, (b) speedup ratio to CPU with particle number for two different parallel algorithms, and (c) comparison of time cost of parallel algorithms PP and PCCP on a single GPU with 500 K spherical particles. </br>

Check for details in paper: </br>
[1].	J. Gan, Z. Zhou, A. Yu, A GPU-based DEM approach for modelling of particulate systems, Powder Technology, 301 (2016) 1172-1182. </br>

Contact Dr. Jieqing Gan at jieqing.gan@monash.edu for more details of the codes.</br>
