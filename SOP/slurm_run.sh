#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -t 00-04:00
#SBATCH -p seas_dgx1
#SBATCH --mem=32768
#SBATCH --gres=gpu:1
#SBATCH -o /n/home00/apalrecha/personal/computational-linear-algebra/SOP/slurm_output/sop_exp_%j.out
#SBATCH -e /n/home00/apalrecha/personal/computational-linear-algebra/SOP/slurm_output/sop_exp_%j.err

module load Anaconda3/5.0.1-fasrc02
cd /n/home00/apalrecha/personal/computational-linear-algebra/SOP/
source activate /n/pfister_lab2/Lab/akash/envs/nbase

echo ""
echo "Begin experiment ..."
python3 run_exp.py
echo "Experiment complete !"
echo ""

