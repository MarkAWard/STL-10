#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -l mem=24GB
#PBS -N ACC
#PBS -M $USER@nyu.edu

echo "Starting job at `date`"
echo

echo "purging module environment"
echo
module purge
module load torch

RUNDIR=$SCRATCH/logs/ACC-${PBS_JOBID/.*}
mkdir -p $RUNDIR
 
cd $RUNDIR

cp $SCRATCH/DeepLearning/STL-10/* .

echo "running the file..."
echo

/scratch/courses/DSGA1008/bin/./th  create_scores.lua -save $RUNDIR

echo "Done"