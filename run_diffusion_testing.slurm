#! /bin/sh

#SBATCH --job-name=adv-ddpo # job name
#SBATCH --output=/home/sharifm/teaching/tml-0368-4075/2023-spring/students/giladd/adv-ddpo/diffusion_testing.out # redirect stdout
#SBATCH --error=/home/sharifm/teaching/tml-0368-4075/2023-spring/students/giladd/adv-ddpo/diffusion_testing.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total
#SBATCH --account=gpu-students

python diffusion_testing.py
