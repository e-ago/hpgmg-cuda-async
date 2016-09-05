#!/bin/bash

#MPI PROC
PROC=2
#LOG2 BOX DIM
SIZE=(4 5 6 7)
#default size >= 4
# 16^3 * 4 = 16384
# 8^3  * 4 = 2048
# 8^3  * 1 = 512
# 4^3  * 1 = 64
# 2^3 * 1 = 8
THRESHOLD=(10000 2000 500 60 0)
MODE=(1 2 3) #MPI, COMM, ASYNC, GPU

echo "===================  TEST HOSTALLOC ===================="
EXCHANGE_HOST_ALLOC=1
EXCHANGE_MALLOC=0

for var_mode in "${MODE[@]}"
do

#Foreach size, try all threshold
for var_size in "${SIZE[@]}"
do
	echo "SIZE($var_size)"
	for var_threshold in "${THRESHOLD[@]}"
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

		$HOME/peersync/src/hpgmg/elerun.sh $PROC $var_comm $var_async $var_gpu $var_size 8 $EXCHANGE_HOST_ALLOC $EXCHANGE_MALLOC $var_threshold &> tmp_out.txt
                echo "Size: $var_size, Threshold: $var_threshold, Mode: $var_print_mode"
		egrep "use cuda" tmp_out.txt
		egrep "time=" tmp_out.txt
	done
done
#close mode
done

echo ""
echo "===================  TEST MALLOC ===================="

EXCHANGE_HOST_ALLOC=0
EXCHANGE_MALLOC=1

for var_mode in "${MODE[@]}"
do

#Foreach size, try all threshold
for var_size in "${SIZE[@]}"
do
        echo "SIZE($var_size)"
        for var_threshold in "${THRESHOLD[@]}"
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

                $HOME/peersync/src/hpgmg/elerun.sh $PROC $var_comm $var_async $var_gpu $var_size 8 $EXCHANGE_HOST_ALLOC $EXCHANGE_MALLOC $var_threshold &> tmp_out.txt
                echo "Size: $var_size, Threshold: $var_threshold, Mode: $var_print_mode"
                egrep "use cuda" tmp_out.txt
                egrep "time=" tmp_out.txt
        done
done
#close mode
done

