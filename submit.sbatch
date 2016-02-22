#!/bin/bash

#SBATCH --job-name=3dLGT
#SBATCH --output=sme.out
#SBATCH --error=sme.err
#SBATCH --time=3:00:00
#SBATCH --qos=iric
#SBATCH --mem=60000
#SBATCH --partition=iric
#SBATCH --mail-type=END
#SBATCH --mail-user=phsun@stanford.edu

#python 3dIsing.py 
python 3dLGT_Z2.py 
