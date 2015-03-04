#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -l walltime=2:00:00
#PBS -l mem=24GB
#PBS -N RESULTS
#PBS -M $USER@nyu.edu

echo "Starting job at `date`"
echo

echo "purging module environment"
echo
module purge
module load torch

RUNDIR=$SCRATCH/logs/predictions-128
mkdir -p $RUNDIR
 
cd $RUNDIR

cp $SCRATCH/DeepLearning/STL-10/* .

echo "running the file..."
echo

/scratch/courses/DSGA1008/bin/./th  test_scores.lua

echo "Done"