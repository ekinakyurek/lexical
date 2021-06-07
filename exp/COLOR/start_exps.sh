#!/bin/sh

# for lamda in 0.05 0.1 0.2 0.5; do
#   for kll in 0.1 0.2 0.5 1.0; do
#       for ent in 0.0 0.00001 0.0001 0.001; do
# 	  screen -X -S lamda_${lamda}_kll_${kll}_ent_${ent} quit
#       done
#   done
# done
#nbatch_1_lr_0.02_Nsample_50_dim_512_lamda_0.05_kll_1.0_ent_0.001
takejob='salloc --gres=gpu:volta:1 --time=24:00:00 --constraint=xeon-g6 --cpus-per-task=5 --qos=high srun'
for nbatch in 1; do # 2 3; do
    for lr in 0.02;do #0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0; do
	for Nsample in 100; do
	    for dim in 512; do
		for lamda in 0.0075; do
		    #for kll in 1.0; do
		    #for ent in 0.001; do
		    #expname=noapprox_${nbatch}_lr_${lr}_Nsample_${Nsample}_dim_${dim}_lamda_${lamda}_kll_${kll}_ent_${ent}
		    expname=noapprox_${nbatch}_color_lamda_${lamda}
		    mkdir -p $expname
		    cd $expname
		    val_steps=$((14 / nbatch))
		    max_steps=$((50*14 / nbatch))
		    tolarance=$max_steps
		    [ -d "run.sh" ] && rm "run.sh"
		    cat > "run.sh" <<EOF
#!/bin/sh		
home="../../"
for i in \`seq 12 19\`
do    
PYTHONHASHSEED=0 python \$home/main.py \\
         --seed \$i \\
         --n_batch ${nbatch} \\
       	 --dim ${dim} \\
	 --lr ${lr} \\
         --regularize \\
         --Nsample ${Nsample} \\
         --lamda ${lamda} \\
         --temp 1.0 \\
         --full_data \\
         --dropout 0.1 \\
	 --accum_count 1 \\
	 --valid_steps ${val_steps} \\
	 --attention \\
	 --max_step ${max_steps} \\
	 --tolarance ${tolarance} \\
	 --warmup_steps 14 \\
	 --noshuffle \\
	 --nobidirectional > eval.\$i.out 2> eval.\$i.err
done
EOF
		    chmod u+x run.sh
		    screen -S ${expname} -d -m bash -c  "$takejob ./run.sh"
		    cd ..
		    #			done
		    #		    done
		done
	    done
	done
    done
done


