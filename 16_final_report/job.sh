#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00
#$ -o output
#$ -e error

. /etc/profile.d/modules.sh

module load cuda/11.2.146
module load gcc
module load openmpi

date
mpirun -n 4 c.out
echo -----
