#!/bin/bash
#SBATCH --job-name=gecacolor
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-1
i=0
for nbatch in 14; do
  for paug in 0.3; do
    for lr in 0.001; do
      for dim in 256; do
        $((i++));
        if [[ $i -eq $SLURM_ARRAY_TASK_ID ]]; then
          expname=geca_nbatch_${nbatch}_dim_${dim}_lr_${lr}_paug_${paug}
          mkdir -p $expname
          cd $expname
          val_steps=$((14 / nbatch))
          max_steps=$((200 * 14 / nbatch))
          tolarance=$max_steps
          home="../../../"
          for i in `seq 0 15`
          do
            PYTHONHASHSEED=0 python $home/main.py \
            --seed $i \
            --n_batch ${nbatch} \
            --n_layers 2 \
            --dim ${dim} \
            --lr ${lr} \
            --nolr_schedule \
            --full_data \
            --dropout 0.3 \
            --accum_count 1 \
            --valid_steps ${val_steps} \
            --attention \
            --max_step ${max_steps} \
            --tolarance ${tolarance} \
            --paug ${paug} \
            --aug_file $home/GECA/COLOR/composed.0.json \
            --noqxy \
            --nobidirectional > eval.$i.out 2> eval.$i.err
          done
          cd ..
        fi
      done
    done
  done
done
