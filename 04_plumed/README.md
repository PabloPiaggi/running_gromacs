# PLUMED

## Compilation and patching

In order to use PLUMED with Gromacs, PLUMED must be compiled first.
Once that PLUMED has been compiled, Gromacs needs to be 'patched'.
This means that some lines of the mdrun executable will be changed in order to interface the two codes.
After 'patching', Gromacs should be compiled as usual.

There are three patching modes, runtime, shared and static.
The preferred one is runtime since it allows to change PLUMED's version simply by changing an environment variable.
In many clusters it is not straightforward to make the runtime patching mode work and in these cases one must use other patching modes.
Fortunately it seems to work properly on tiger and traverse (della?).

## Running Gromacs and PLUMED

If Gromacs was patched in runtime mode then the desired version of PLUMED must be sourced (see an example below).
Afterwards, the flag -plumed should be used to call PLUMED from Gromacs.
Many parts of PLUMED are paralellized using MPI and openMP.
Typically, PLUMED will use the same number of MPI processes and openMP threads as Gromacs is using.
PLUMED cannot take advantage of the GPUs.

## TigerGPU

This is a compilation script:

```bash
#!/bin/bash

module purge
module load intel/19.0/64/19.0.1.144
module load intel-mpi/intel/2019.1/64
module load rh/devtoolset/7

#############################################################
# PLUMED
#############################################################

plumedversion=v2.6
git clone -b ${plumedversion} https://github.com/plumed/plumed2 plumed2-${plumedversion}
cd plumed2-${plumedversion}

#############################################################
# starting build of plumed
#############################################################

OPTFLAGS="-Ofast -xCORE-AVX2 -mtune=broadwell -DNDEBUG"
./configure --enable-modules=all CXX=mpiicpc CXXFLAGS="$OPTFLAGS"
make -j 10
source sourceme.sh

cd ../

#############################################################
# GROMACS
#############################################################

version=2019.4
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-${version}.tar.gz
tar -zxvf gromacs-${version}.tar.gz
cd gromacs-${version}
plumed patch -p -e gromacs-2019.4 --runtime
mkdir build_stage1
cd build_stage1

#############################################################
# starting build of gmx (stage 1)
#############################################################

OPTFLAGS="-Ofast -xCORE-AVX2 -mtune=broadwell -DNDEBUG"

cmake3 .. -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=icc -DCMAKE_C_FLAGS_RELEASE="$OPTFLAGS" \
-DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS_RELEASE="$OPTFLAGS" \
-DGMX_BUILD_MDRUN_ONLY=OFF -DGMX_MPI=OFF -DGMX_OPENMP=ON \
-DGMX_SIMD=AVX2_256 -DGMX_DOUBLE=OFF \
-DGMX_FFT_LIBRARY=mkl \
-DGMX_GPU=OFF \
-DCMAKE_INSTALL_PREFIX=$HOME/.local \
-DGMX_COOL_QUOTES=OFF -DREGRESSIONTEST_DOWNLOAD=ON

make -j 10
make check
make install
cd ..

#############################################################
# starting build of mdrun_mpi (stage 2)
#############################################################

mkdir build_stage2
cd build_stage2

module load cudatoolkit/10.1

cmake3 .. -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=icc -DCMAKE_C_FLAGS_RELEASE="$OPTFLAGS" \
-DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS_RELEASE="$OPTFLAGS" \
-DGMX_BUILD_MDRUN_ONLY=ON -DGMX_MPI=ON -DGMX_OPENMP=ON \
-DGMX_SIMD=AVX2_256 -DGMX_DOUBLE=OFF \
-DGMX_FFT_LIBRARY=mkl \
-DGMX_GPU=ON -DGMX_CUDA_TARGET_SM=60 \
-DCMAKE_INSTALL_PREFIX=$HOME/.local \
-DGMX_COOL_QUOTES=OFF -DREGRESSIONTEST_DOWNLOAD=ON
#-DCUDA_NVCC_FLAGS_RELEASE="-ccbin=icpc -O3 --use_fast_math -arch=sm_60 --gpu-code=sm_60"

make -j 10
source ../build_stage1/scripts/GMXRC
tests/regressiontests-${version}/gmxtest.pl all
make install
```

Below is a sample Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=gmx-plumed    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=<YourNetID>@princeton.edu

module purge
module load intel/19.0/64/19.0.1.144
module load intel-mpi/intel/2019.1/64
module load rh/devtoolset/7
module load cudatoolkit/10.1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PLUMED_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GMX_MAXBACKUP=-1

# Source plumed using for instance the following lines
# source ~/installation-path/plumed2-v2.6/sourceme.sh

srun mdrun_mpi -s topol.tpr -gpu_id 0 -maxh 1 -ntomp ${OMP_NUM_THREADS} -plumed plumed.dat
```

## Traverse

To compile Gromacs and PLUMED on Traverse you can use the script below:

```bash
#!/bin/bash
version_fftw=3.3.8
version_gmx=2019.4
version_plumed=v2.6

#############################################################
# build a fast version of FFTW
#############################################################
wget ftp://ftp.fftw.org/pub/fftw/fftw-${version_fftw}.tar.gz
tar -zxvf fftw-${version_fftw}.tar.gz
cd fftw-${version_fftw}

module purge
module load rh/devtoolset/8

./configure CC=gcc CFLAGS="-Ofast -mcpu=power9 -mtune=power9 -DNDEBUG" --prefix=$HOME/.local \
--enable-shared --enable-single --enable-vsx --disable-fortran

make
make install
cd ..

#############################################################
# build plumed
#############################################################

module purge
module load rh/devtoolset/7
module load openmpi/devtoolset-8/4.0.1/64

git clone -b ${version_plumed} https://github.com/plumed/plumed2 plumed2-${version_plumed}
cd plumed2-${version_plumed}
OPTFLAGS="-Ofast -mcpu=power9 -mtune=power9 -mvsx -DNDEBUG"
./configure --enable-modules=all CXXFLAGS="$OPTFLAGS"
make -j 10
source sourceme.sh
cd ../

#############################################################
# build gmx (for single node jobs)
#############################################################
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-${version_gmx}.tar.gz
tar -zxf gromacs-${version_gmx}.tar.gz
cd gromacs-${version_gmx}
plumed patch -p -e gromacs-2019.4 --runtime
mkdir build_stage1
cd build_stage1

module load cudatoolkit/10.2

OPTFLAGS="-Ofast -mcpu=power9 -mtune=power9 -mvsx -DNDEBUG"

cmake3 .. -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=gcc -DCMAKE_C_FLAGS_RELEASE="$OPTFLAGS" \
-DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS_RELEASE="$OPTFLAGS" \
-DGMX_BUILD_MDRUN_ONLY=OFF -DGMX_MPI=OFF -DGMX_OPENMP=ON \
-DGMX_SIMD=IBM_VSX -DGMX_DOUBLE=OFF \
-DGMX_FFT_LIBRARY=fftw3 \
-DFFTWF_LIBRARY=$HOME/.local/include \
-DFFTWF_LIBRARY=$HOME/.local/lib/libfftw3f.so \
-DGMX_GPU=ON -DGMX_CUDA_TARGET_SM=70 \
-DGMX_EXTERNAL_BLAS=ON -DGMX_BLAS_USER=/usr/lib64/libessl.so \
-DGMX_EXTERNAL_LAPACK=ON -DGMX_LAPACK_USER=/usr/lib64/libessl.so \
-DCMAKE_INSTALL_PREFIX=$HOME/.local \
-DGMX_COOL_QUOTES=OFF -DREGRESSIONTEST_DOWNLOAD=ON

make -j 10
make check
make install

#############################################################
# build mdrun_mpi (for multi-node jobs)
#############################################################
mkdir build_stage2
cd build_stage2

cmake3 .. -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=gcc -DCMAKE_C_FLAGS_RELEASE="$OPTFLAGS" \
-DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS_RELEASE="$OPTFLAGS" \
-DGMX_BUILD_MDRUN_ONLY=ON -DGMX_MPI=ON -DGMX_OPENMP=ON \
-DGMX_SIMD=IBM_VSX -DGMX_DOUBLE=OFF \
-DGMX_FFT_LIBRARY=fftw3 \
-DFFTWF_LIBRARY=$HOME/.local/include \
-DFFTWF_LIBRARY=$HOME/.local/lib/libfftw3f.so \
-DGMX_GPU=ON -DGMX_CUDA_TARGET_SM=70 \
-DGMX_EXTERNAL_BLAS=ON -DGMX_BLAS_USER=/usr/lib64/libessl.so \
-DGMX_EXTERNAL_LAPACK=ON -DGMX_LAPACK_USER=/usr/lib64/libessl.so \
-DCMAKE_INSTALL_PREFIX=$HOME/.local \
-DGMX_COOL_QUOTES=OFF -DREGRESSIONTEST_DOWNLOAD=ON

make -j 10
source ../build_stage1/scripts/GMXRC
tests/regressiontests-${version_gmx}/gmxtest.pl all
make install
```

## Benchmarks

### Traverse

#### 432 TIP4P/Ice molecules + Coordination number

| # GPU | # MPI processes | # openMP threads | Performance (ns/day) | Performance without PLUMED (ns/day) | Ratio |
|-------|-----------------|------------------|----------------------|-------------------------------------|-------|
| 0     | 0               | 4                | 34.614               | 66.927                              | 0.52  |
| 0     | 0               | 8                | 87.599               | 64.985                              | 1.35  |
| 0     | 0               | 16               | 91.27                | 244.788                             | 0.37  |
| 1     | 1               | 1                | 97.991               | 379.507                             | 0.26  |
| 1     | 1               | 2                | 111.211              | 374.918                             | 0.30  |
| 1     | 1               | 4                | 113.256              | 283.378                             | 0.40  |
| 1     | 1               | 8                | 136.639              | 277.121                             | 0.49  |
| 1     | 1               | 16               | 147.813              | 303.097                             | 0.49  |
| 1     | 4               | 1                | 95.61                | 173.969                             | 0.55  |

Conclusions:
- Best performance with and without plumed corresponds to different configurations!					
- PLUMED takes more or less half of the total time					
- The performance of both codes is relevant					
- Best configurations use GPUs!					

#### 432 TIP4P/Ice molecules + Q6

| # GPU | # MPI processes | # openMP threads | Performance (ns/day) | Performance without PLUMED (ns/day) | Ratio |
|-------|-----------------|------------------|----------------------|-------------------------------------|-------|
| 0     | 0               | 4                |                      | 66.927                              | 0.00  |
| 0     | 0               | 8                |                      | 64.985                              | 0.00  |
| 0     | 0               | 16               |                      | 244.788                             | 0.00  |
| 1     | 1               | 1                | 22.693               | 379.507                             | 0.06  |
| 1     | 1               | 2                | 26.963               | 374.918                             | 0.07  |
| 1     | 1               | 4                | 23.914               | 283.378                             | 0.08  |
| 1     | 1               | 8                |                      | 277.121                             | 0.00  |
| 1     | 1               | 16               | 34.999               | 303.097                             | 0.12  |
| 1     | 4               | 1                | 27.931               | 173.969                             | 0.16  |				

Conclusions:
- Performance limited by PLUMED
- PLUMED takes 90% of the calculation time
- Performance of Gromacs is irrelevant, for instance using 4 openMP threads for Gromacs is better than 4 MPI processes. However this doesn't show on the Gromacs+Plumed performance because for PLUMED this configurations give the same speed.
- The advantage of using GPUs is limited.
