#!/bin/bash
#SBATCH -J my_cpu_job              # Job name (shown in squeue and emails)
#SBATCH -o logs/my_cpu_job_%j.out  # stdout; %j = job ID so runs don't overwrite
#SBATCH -e logs/my_cpu_job_%j.err  # stderr, kept separate from stdout
#SBATCH -p gg                      # Vista Grace-Grace CPU partition 
#SBATCH -N 1                       # one node and all cores
#SBATCH -n 1                       # one task (not MPI); parallelism comes from multiprocessing
#SBATCH -t 08:00:00                # wall-clock limit (HH:MM:SS); job is killed past this
#SBATCH -A AST24021                # allocation to charge
#SBATCH --mail-type=END,FAIL       # email on completion or failure only
#SBATCH --mail-user=dc53555@eid.utexas.edu # use an address that actually receives mail

set -euo pipefail                  # stop on error, unset vars, and mid-pipe failures
mkdir -p logs                      # ensure the -o/-e log dir exists

cd "$SLURM_SUBMIT_DIR"             # start where sbatch was run, so relative paths resolve

# --- environment ---
source /scratch/11648/daracheng4787/myenv/bin/activate   # activate the venv

# keep each worker's BLAS single-threaded
# so N workers don't each spawn N threads and oversubscribe the 144 cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host=$(hostname)"            # log the node, for debugging
echo "cores=$(nproc)"             # log core count actually available (144 on gg)

# --- run ---
python tracer_pipeline.py                # your CPU-bound work; size the Pool to os.cpu_count()