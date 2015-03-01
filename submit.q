#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -l mem=24GB
#PBS -N LEM
#PBS -M maw627@nyu.edu

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
cp $HOME/DeepLearning/STL-10/codeBase.lua $RUNDIR
cd $RUNDIR


echo "running the file..."
echo
th codeBase.lua

echo "Done"
 