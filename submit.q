#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -l mem=24GB
#PBS -N LEM
#PBS -M $USER@nyu.edu
#PBS -j oe

cd $HOME/DeepLearning/STL-10

echo "job starting on `date`"
echo

echo "purging module environment"
echo
module purge

echo "loading modules..."
echo
module load cuda/6.5.12
module load torch


RUNDIR=$SCRATCH/logs/A2-${PBS_JOBID/.*}
mkdir -p $RUNDIR
 
cd $RUNDIR

cp $SCRATCH/DeepLearning/STL-10/* .

echo "running the file..."
echo
/scratch/courses/DSGA1008/bin/./th codeBase.lua -epochs 1000 -type double -batchSize 8

echo "Done"
 