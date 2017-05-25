#!/usr/bin/env bash

#MPI PROC
PROC=(2 4 8 16)
#LOG2 BOX DIM
SIZE=(4 5 6 7)
#default size >= 4
# 16^3 * 4 = 16384
# 8^3  * 4 = 2048
# 8^3  * 1 = 512
# 4^3  * 1 = 64
# 2^3 * 1 = 8
MODE=(0 1 2 3) #MPI, COMM, ASYNC, GPU

for var_mode in "${MODE[@]}"
do
        for var_size in "${SIZE[@]}"
        do
        	for var_proc in "${PROC[@]}"
        	do
                        for num_iter in {1..5}
                        do
                		if [ $var_mode -eq 0 ]; then
                			var_print_mode="MPI"
                			var_comm=0;
                                	var_async=0;
                                	var_gpu=0;
                                elif [ $var_mode -eq 1 ]; then
                                        var_print_mode="COMM"
                                        var_comm=1;
                                        var_async=0;
                                        var_gpu=0;
                		elif [ $var_mode -eq 2 ]; then
                			var_print_mode="ASYNC"		
                                        var_comm=1;
                                        var_async=1;
                                        var_gpu=0;
                		else
                			var_print_mode="GPU-initiated"
                                        var_comm=1;
                                        var_async=1;
                                        var_gpu=1;
                		fi
                		echo "MODE: $var_print_mode, SIZE: $var_size, PROC: $var_proc, ITER: $num_iter"
                		file_out="hpgmg-$var_print_mode-s$var_size-p$var_proc.txt"
                		./run.sh $var_proc $var_comm $var_async $var_gpu $var_size 8 &> $file_out
                		egrep "use cuda" $file_out
                		egrep "Total by level" $file_out
                	done
                done
        done
done