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

TIME_LOG=execution_times.txt

echo "Execution Times" > $TIME_LOG
echo "==============================" >> $TIME_LOG

for i in $(seq 7 12)
do
    echo "Running classic vlen=$i"

    /usr/bin/time -f "classic vlen=$i : real=%e user=%U sys=%S" \
    -o $TIME_LOG -a \
    srun apptainer exec $GEM5_WORKSPACE/gem5.sif \
    $GEM5_PATH/gem5.opt \
    --outdir=./scaled_dot/classic/classic_vlen$i \
    ./cpu_benchmark.py \
    --vlen=$i \
    --file_path=./workload/scaled_dot_product/scaled_dot_product.bin

    echo "Running vectorized vlen=$i"

    /usr/bin/time -f "vectorized vlen=$i : real=%e user=%U sys=%S" \
    -o $TIME_LOG -a \
    srun apptainer exec $GEM5_WORKSPACE/gem5.sif \
    $GEM5_PATH/gem5.opt \
    --outdir=./scaled_dot/vectorized/vectorized_vlen$i \
    ./cpu_benchmark.py \
    --vlen=$i \
    --file_path=./workload/scaled_dot_product/scaled_dot_product_vectorized.bin
done