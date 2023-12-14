#!/bin/bash
#SBATCH -p gpu22
#SBATCH --mem=250G
#SBATCH --gres gpu:1
#SBATCH -t 48:00:00
#SBATCH -a 1-8%8
#SBATCH -o ./logs/main-%j-%a.out

 
eval "$(conda shell.bash hook)"
eval "$(ulimit -Sn unlimited)"
  
conda activate torch112

n_workers=8
generation_mode='test'


case $SLURM_ARRAY_TASK_ID in
    1)
        N_WORKERS=$n_workers WORKER_ID=0 GENERATION_MODE=$generation_mode python main.py  &
        ;;
    2) 
        N_WORKERS=$n_workers WORKER_ID=1 GENERATION_MODE=$generation_mode python main.py  &
        ;;
    3)  
        N_WORKERS=$n_workers WORKER_ID=2 GENERATION_MODE=$generation_mode python main.py &
        ;;
    4)  
        N_WORKERS=$n_workers WORKER_ID=3 GENERATION_MODE=$generation_mode python main.py &
        ;;

    5)  
        N_WORKERS=$n_workers WORKER_ID=4 GENERATION_MODE=$generation_mode python main.py &
        ;;
    6)  
        N_WORKERS=$n_workers WORKER_ID=5 GENERATION_MODE=$generation_mode python main.py &
        ;;
    7)  
        N_WORKERS=$n_workers WORKER_ID=6 GENERATION_MODE=$generation_mode python main.py &
        ;;
    8)  
        N_WORKERS=$n_workers WORKER_ID=7 GENERATION_MODE=$generation_mode python main.py &
        ;;
esac
wait