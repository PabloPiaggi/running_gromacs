# PLUMED

## Compilation and patching

In order to use PLUMED with Gromacs, PLUMED must be compiled first.
Once that PLUMED has been compiled, Gromacs needs to be 'patched'.
This means that some lines of the mdrun executable will be changed in order to interface the two codes.
After 'patching' Gromacs it should be compiled as usual.

There are three patching modes, runtime, shared and static.
The preferred one is runtime since it allows to change PLUMED version simply by changing an environment variable.
In many clusters it is not straightforward to make the runtime patch work.
Fortunately it seems to work properly on tiger.

This is a compilation script:

## TigerGPU

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

The flag -plumed should be used to call PLUMED from Gromacs.
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
