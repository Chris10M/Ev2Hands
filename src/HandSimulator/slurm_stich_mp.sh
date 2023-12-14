#!/bin/bash
#SBATCH -p cpu20
#SBATCH -t 48:00:00
#SBATCH -a 1-3%3
#SBATCH -o ./logs/stich_mp-%j-%a.out

 
eval "$(conda shell.bash hook)"
eval "$(ulimit -Sn unlimited)"
  
conda activate torch112

case $SLURM_ARRAY_TASK_ID in
    1)
        GENERATION_MODE='train' python stich_mp.py  &
        ;;
    2) 
        GENERATION_MODE='test' python stich_mp.py  &
        ;;
    3)  
        GENERATION_MODE='val' python stich_mp.py &
        ;;

esac
wait
