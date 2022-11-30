#!/bin/bash
#SBATCH --job-name=IA_GPU_04                    # nombre del job
#SBATCH --nodes=1                               # cantidad de nodos
#SBATCH --ntasks=1                              # cantidad de tareas
#SBATCH --cpus-per-task=8                       # cpu-cores por task 
#SBATCH --mem=32G                               # memoria total por nodo
#SBATCH --gres=gpu:2                            # numero de gpus por nodo
#SBATCH --output=../outputs/IA_GPU.out          # archivo de salida
#SBATCH --error=../outputs/IA_CPU.err           # archivo de error

module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate BT

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -W ignore main.py

conda deactivate