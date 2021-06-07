takejob='salloc --gres=gpu:volta:1 --time=24:00:00 --job-name=baseline --constraint=xeon-g6 --cpus-per-task=5 --qos=high  srun'

for lr in 1.0; do
    for warmup_steps in 4001; do #orig: 4000
        for max_steps in 8000; do #orig 30000
    expname=baseline_aligner_TRANSLATE_lr_${lr}_warmup_${warmup_steps}_max_${max_steps}
    mkdir -p $expname
    cd $expname
    [ -d "run.sh" ] && rm "run.sh"
    cat > "run.sh" <<EOF
#!/bin/sh
home="../../../"
for i in 0 1 2 3 4
do
python -u  \$home/main.py \\
	--seed \$i \\
	--n_batch 128 \\
	--n_layers 2 \\
	--noregularize \\
        --dim 512 \\
        --lr ${lr} \\
        --temp 1.0 \\
        --dropout 0.4 \\
        --beam_size 5 \\
	--gclip 5.0 \\
	--accum_count 4 \\
	--valid_steps 500 \\
        --warmup_steps ${warmup_steps} \\
	--max_step ${max_steps} \\
	--tolarance 10 \\
	--tb_dir ${expname} \\
	--TRANSLATE > translate.eval.\$i.out 2> translate.eval.\$i.err
done
EOF
            chmod u+x run.sh
            screen -S ${expname} -d -m bash -c  "$takejob ./run.sh"
            cd ..
        done
    done
done
