#!/bin/bash
# example: ./clean.sh MTSD 1 gt_reduced_10_HFLP.train.txt
EXP=$2
DATASET=$1
GT=$3

delete_file()
{
	path=$1
	echo Deleting ${path}
	if [ -d $path ]; then
		echo "dir exists"
		rm -rf ${path}
	else
	
	#if [ ! -d $path ]; then
		echo "dir doesn't exists"
	fi
	#rm -rf ${path}
}


delete_file results/${DATASET}_${GT}_result${EXP}/

delete_file data/${DATASET}_${GT}_test${EXP}/

delete_file ../Datasets/${DATASET}_${GT}_AUG${EXP}/

#echo Deleting ${path}
#rm -rf ${path}


#rm -rf data/MTSD_test1/
#rm -rf ../Datasets/GTSDB_AUG1/
#echo results/${DATASET}_result${EXP}/
