#!/bin/bash
#----------------------------------------------------------------------
# SLURM job array: a CHUNK of clusters per array task, one node per task.
# SLURM does the dynamic dispatch — as a node frees up, the next queued
# array task starts, so nodes are kept busy without any manager code.
# Inside each task, cluster_pipeline.py processes its 8 clusters in sequence,
# each using the whole node via ProcessPoolExecutor (multiprocessing over intervals).
#
# 128 clusters, 8 per task -> 16 tasks (indices 0..15). Each cluster ~8 min,
# so ~64 min/task (well above Vista's 15-min minimum charge). %16 runs all
# 16 tasks at once. Confirm the cluster count with:
#     python cluster_pipeline.py --list
#
# Submit:   sbatch run_clusters.slurm
# Requeue only failed tasks later:  sbatch --array=3,7 run_clusters.slurm
#----------------------------------------------------------------------
#SBATCH -J drg_clusters
#SBATCH -o logs/drg_%A_%a.out          # %A=jobid  %a=array index
#SBATCH -e logs/drg_%A_%a.err
#SBATCH -p gg                          # Vista Grace-Grace CPU partition (adjust if needed)
#SBATCH -N 1                           # one node per task
#SBATCH -n 1                           # one launching task; Python fans out to all cores
#SBATCH -t 02:00:00                    
#SBATCH --array=0-15%16                # 16 tasks (8 clusters each), all 16 concurrent
#SBATCH -A AST24021

####SBATCH --mail-type=all
####SBATCH --mail-user=dc53555@eid.utexas.edu

set -euo pipefail
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

# --- environment: activate the venv that has numpy/scipy/matplotlib/pandas/networkx ---
source /scratch/11648/daracheng4787/myenv/bin/activate 

# Keep math libs single-threaded per worker so the process pool (not BLAS) owns the cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Optionally cap the pool if memory-bound (default = all cores):
# export N_WORKERS=72

echo "host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID} cores=$(nproc)"

python cluster_pipeline.py --task "${SLURM_ARRAY_TASK_ID}" --per-task 8
