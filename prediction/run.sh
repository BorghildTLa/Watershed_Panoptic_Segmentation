#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7000mb
#SBATCH --partition=gpu
#SBATCH --gpus=geforce
#SBATCH --time=72:00:00
#SBATCH --output=hail.out
#SBATCH --job-name="PAN-DL2"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module load python/3.6.5
module load tensorflow/1.14.0
module use /home/nlucarelli/privatemodules
module load openslide/3.4.0
module load joblib/0.11
module load imgaug/0.4.0
module load imageio/2.3.0
module list
which python

echo "Launch job"
python3 segmentation_school.py --option predict --project project_name/ --base_dir /blue/pinaki.sarder/username/folder --wsi_ext '.svs,.ndpi,.scn' --classNum 6 --one_network True
#
echo "All Done!"
