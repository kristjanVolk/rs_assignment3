#!/bin/sh
#SBATCH --job-name=gem5_simulation
#SBATCH --output=gem5_log.txt
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --reservation=fri

GEM5_WORKSPACE=/d/hpc/projects/FRI/GEM5/gem5_workspace
GEM5_ROOT=$GEM5_WORKSPACE/gem5
GEM5_PATH=$GEM5_ROOT/build/RISCV_ALL_RUBY

for i in $(seq 7 12)
do
    srun apptainer exec $GEM5_WORKSPACE/gem5.sif $GEM5_PATH/gem5.opt --outdir=./heat_stencil/heat_stencil$i ./cpu_benchmark.py --vlen=$i --file_path=./workload/heat_stencil/heat_stencil.bin
    srun apptainer exec $GEM5_WORKSPACE/gem5.sif $GEM5_PATH/gem5.opt --outdir=./heat_stencil/heat_stencil64KiB_$i ./cpu_benchmark.py --cache_size=64KiB --vlen=$i --file_path=./workload/heat_stencil/heat_stencil.bin
done

