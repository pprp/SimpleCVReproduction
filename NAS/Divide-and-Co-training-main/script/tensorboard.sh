#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/lib
export CUDA_VISIBLE_DEVICES="0"
# run the tensorboard command periodly
echo ""
echo "1 -- splitnet directory"
echo -n "choose the directory: "
read dir_choose

echo ""
echo -n "input the port:"
read port

# set the logdir
case ${dir_choose} in
	1 )
		logdir="${HOME}/models/splitnet"
		;;
	* )
		echo "The choice of thedirectory is illegal!"
		exit 1
		;;
esac


# sleep time, hours
sleep_t=6
times=0

# while loop
while true
do
	# https://stackoverflow.com/questions/40106949/unable-to-open-tensorboard-in-browser
	tensorboard --bind_all --logdir=${logdir} --port=${port} &
	last_pid=$!
	sleep ${sleep_t}h
	kill -9 ${last_pid}
	times=`expr ${times} + 1`
	echo "Restart tensorboard ${times} times."
done

echo "tensorboard is stopped!"
