# Installation

## TigerGPU

The code can be built with:

```
$ ssh <NetID>@tigergpu.princeton.edu
$ cd </path/to/software/directory>  # e.g., cd ~/software
$ wget <https://github.com/PrincetonUniversity/running_gromacs/tree/master/02_install/tigerGpu.sh>
# make modifications to tigerGpu.sh if needed
$ bash tigerGpu.sh | tee build.log
```

The installation script is shown below (follow the directions above -- you do not need to copy this):

```bash
#!/bin/bash
version=2019.4
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-${version}.tar.gz
tar -zxvf gromacs-${version}.tar.gz
cd gromacs-${version}
mkdir build_stage1
cd build_stage1

#############################################################
# starting build of gmx (stage 1)
#############################################################

module purge
module load intel/19.0/64/19.0.1.144

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

#############################################################
# starting build of mdrun_mpi (stage 2)
#############################################################

cd ..
mkdir build_stage2
cd build_stage2

module load intel-mpi/intel/2019.1/64
module load cudatoolkit/10.1
module load rh/devtoolset/7

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
#SBATCH --job-name=gmx           # create a short name for your job
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
module load cudatoolkit/10.1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GMX_MAXBACKUP=-1

BCH=../gpu_bench/rnase_cubic
srun gmx grompp -f $BCH/pme_verlet.mdp -c $BCH/conf.gro -p $BCH/topol.top -o bench.tpr
srun mdrun_mpi -ntomp $SLURM_CPUS_PER_TASK -s bench.tpr -c conf.gro
```

## Traverse

Note that `rh/devtoolset/8` cannot be used to compile Gromacs.

```bash
#!/bin/bash
version_fftw=3.3.8
version_gmx=2019.4

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
# build gmx (for single node jobs)
#############################################################
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-${version_gmx}.tar.gz
tar -zxvf gromacs-${version_gmx}.tar.gz
cd gromacs-${version_gmx}
mkdir build_stage1
cd build_stage1

module purge
module load rh/devtoolset/7
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
cd ..
mkdir build_stage2
cd build_stage2

module load openmpi/devtoolset-8/4.0.1/64

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

Notes:

1. Reference Slurm variables instead of introducing duplication: `SLURM_CPUS_PER_TASK`, `SLURM_NTASKS`, `SLURM_NODES`
see `env | grep SLURM | sort`
2. Gromacs only uses serial FFT's so one does not need an MPI-enabled FFT module
3. use srun in favor of mpirun -np 16

Questions:

1. Thread MPI vs MPI for a single node


## Della

Della is good for single node jobs. You should not be running small jobs on Tiger.

```bash
#!/bin/bash
version=2019.4
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-${version}.tar.gz
tar -zxvf gromacs-${version}.tar.gz
cd gromacs-${version}
mkdir build_stage1
cd build_stage1

#############################################################
# starting build of gmx (stage 1)
#############################################################

module purge
module load intel/19.0/64/19.0.1.144

OPTFLAGS="-Ofast -xCORE-AVX2 -axCORE-AVX512 -DNDEBUG"

cmake3 .. -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=icc -DCMAKE_C_FLAGS_RELEASE="$OPTFLAGS" \
-DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS_RELEASE="$OPTFLAGS" \
-DGMX_BUILD_MDRUN_ONLY=OFF -DGMX_MPI=OFF -DGMX_OPENMP=ON \
-DGMX_DOUBLE=OFF \
-DGMX_FFT_LIBRARY=mkl \
-DGMX_GPU=OFF \
-DCMAKE_INSTALL_PREFIX=$HOME/.local \
-DGMX_COOL_QUOTES=OFF -DREGRESSIONTEST_DOWNLOAD=ON

make -j 10
make check
make install

#############################################################
# starting build of mdrun_mpi (stage 2)
#############################################################

cd ..
mkdir build_stage2
cd build_stage2

module load intel-mpi/intel/2018.3/64

cmake3 .. -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=icc -DCMAKE_C_FLAGS_RELEASE="$OPTFLAGS" \
-DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS_RELEASE="$OPTFLAGS" \
-DGMX_BUILD_MDRUN_ONLY=ON -DGMX_MPI=ON -DGMX_OPENMP=ON \
-DGMX_DOUBLE=OFF \
-DGMX_FFT_LIBRARY=mkl \
-DGMX_GPU=OFF \
-DCMAKE_INSTALL_PREFIX=$HOME/.local \
-DGMX_COOL_QUOTES=OFF -DREGRESSIONTEST_DOWNLOAD=ON

make -j 10
source ../build_stage1/scripts/GMXRC
tests/regressiontests-${version}/gmxtest.pl all
make install
```

Make two versions for Della: avx2 and avx512 if results say so;
#SBATCH --exclude=della-r4c1n[1-16]

If you ever need to run on one of the cascade lake nodes, you can create an exclusion list of all the other nodes:

```
snodes | grep -v 'cascade\|HOSTNAMES' | cut -c 1-13 | tr -s '\n' ',' | tr -d '[:blank:]' > exclude.nodes
```

## Perseus

```bash
#!/bin/bash
version=2019.4
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-${version}.tar.gz
tar -zxvf gromacs-${version}.tar.gz
cd gromacs-${version}
mkdir build_stage1
cd build_stage1

#############################################################
# starting build of gmx (stage 1)
#############################################################

module purge
module load intel/19.0/64/19.0.1.144

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

#############################################################
# starting build of mdrun_mpi (stage 2)
#############################################################

cd ..
mkdir build_stage2
cd build_stage2

module load intel-mpi/intel/2018.3/64

cmake3 .. -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=icc -DCMAKE_C_FLAGS_RELEASE="$OPTFLAGS" \
-DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS_RELEASE="$OPTFLAGS" \
-DGMX_BUILD_MDRUN_ONLY=ON -DGMX_MPI=ON -DGMX_OPENMP=ON \
-DGMX_SIMD=AVX2_256 -DGMX_DOUBLE=OFF \
-DGMX_FFT_LIBRARY=mkl \
-DGMX_GPU=OFF \
-DCMAKE_INSTALL_PREFIX=$HOME/.local \
-DGMX_COOL_QUOTES=OFF -DREGRESSIONTEST_DOWNLOAD=ON

make -j 10
source ../build_stage1/scripts/GMXRC
tests/regressiontests-${version}/gmxtest.pl all
make install
```

Note that the latest Intel compiler cannot be used:

```
cmake fails with when using intel/19.0.5
-- Checking for 64-bit off_t
*** Error in `/opt/intel/compilers_and_libraries_2019.5.281/linux/bin/intel64/icc': free(): invalid next size (fast): 0x00002addf0000970 ***
```

## TigerCPU

## Adroit (GPU)

Gromacs can be built for GPU use on Adroit by running the following commands:

```
$ wget
$ bash adroit.sh | tee build.log
```

The `tee` command is used to have the output go to the terminal as well as a file.

The installation script is reproduced below:

```bash
#!/bin/bash
version=2019.4
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-${version}.tar.gz
tar -zxvf gromacs-${version}.tar.gz
cd gromacs-${version}
mkdir build_stage1
cd build_stage1

#############################################################
# starting build of gmx (stage 1)
#############################################################

module purge
module load intel/19.0/64/19.0.1.144

OPTFLAGS="-Ofast -xCORE-AVX512 -mtune=skylake-avx512 -DNDEBUG"

cmake3 .. -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=icc -DCMAKE_C_FLAGS_RELEASE="$OPTFLAGS" \
-DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS_RELEASE="$OPTFLAGS" \
-DGMX_BUILD_MDRUN_ONLY=OFF -DGMX_MPI=OFF -DGMX_OPENMP=ON \
-DGMX_SIMD=AVX_512 -DGMX_DOUBLE=OFF \
-DGMX_FFT_LIBRARY=mkl \
-DGMX_GPU=OFF \
-DCMAKE_INSTALL_PREFIX=$HOME/.local \
-DGMX_COOL_QUOTES=OFF -DREGRESSIONTEST_DOWNLOAD=ON

make -j 10
make check
make install

#############################################################
# starting build of mdrun_mpi (stage 2)
#############################################################

cd ..
mkdir build_stage2
cd build_stage2

module load intel-mpi/intel/2018.3/64
module load cudatoolkit/10.1
module load rh/devtoolset/7

cmake3 .. -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=icc -DCMAKE_C_FLAGS_RELEASE="$OPTFLAGS" \
-DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS_RELEASE="$OPTFLAGS" \
-DGMX_BUILD_MDRUN_ONLY=ON -DGMX_MPI=ON -DGMX_OPENMP=ON \
-DGMX_SIMD=AVX_512 -DGMX_DOUBLE=OFF \
-DGMX_FFT_LIBRARY=mkl \
-DGMX_GPU=ON -DGMX_CUDA_TARGET_SM=70 \
-DCMAKE_INSTALL_PREFIX=$HOME/.local \
-DGMX_COOL_QUOTES=OFF -DREGRESSIONTEST_DOWNLOAD=ON

make -j 10
source ../build_stage1/scripts/GMXRC
tests/regressiontests-${version}/gmxtest.pl all
make install
```
