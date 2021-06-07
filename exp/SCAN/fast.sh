#!/bin/bash
#SBATCH --job-name=fastscan
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-20

lr=1.0
align="intersect"
warmup_steps=4000
max_steps=8000
batch_size=128
home="../../../"
k=0
for split in around_right jump; do
        if [[ $split = jump ]]; then
                split_folder="add_prim_split"
        else
                split_folder="template_split"
        fi
        expname=${split}_fast_${align}
        mkdir -p $expname
        for i in `seq 0 9`
        do
                $(( k++ ))
                if [[ $k -eq $SLURM_ARRAY_TASK_ID ]]; then
                        cd $expname
                        python -u  $home/main.py \
                        --seed $i \
                        --n_batch ${batch_size} \
                        --n_layers 2 \
                        --noregularize \
                        --dim 512 \
                        --lr ${lr} \
                        --temp 1.0 \
                        --dropout 0.4 \
                        --beam_size 5 \
                        --gclip 5.0 \
                        --accum_count 4 \
                        --valid_steps 500 \
                        --warmup_steps ${warmup_steps} \
                        --max_step ${max_steps} \
                        --tolarance 10 \
                        --copy \
                        --scan_split ${split} \
                        --highdrop \
                        --tb_dir ${expname} \
                        --aligner $home/SCAN/${split_folder}/alignments/${align}.align.o.json \
                        --SCAN > eval.$i.out 2> eval.$i.err
                fi
        done
done
