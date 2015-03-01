#!/bin/bash
#PBS -l nodes=1:ppn=4:gpus=4:titan
#PBS -l walltime=10:00:00
#PBS -l mem=24GB
#PBS -N LEM
#PBS -M $USER@nyu.edu
#PBS -j oe

echo "job starting on `date`"
echo

echo "purging module environment"
echo
module purge

echo "loading modules..."
echo
module load cuda/6.5.12
module load torch

echo "setting up files and directories..."
echo
RUNDIR=$SCRATCH/logs/A2-${PBS_JOBID/.*}
mkdir -p $RUNDIR
cd $RUNDIR

echo "running the file..."
echo
th $SCRATCH/DeepLearning/STL-10/codeBase.lua -type cuda

echo "Done"
 