#!/bin/bash

#SBATCH --job-name=3d_16
#SBATCH --output=sme.out
#SBATCH --error=sme.err
#SBATCH --time=20:00:00
#SBATCH --qos=iric
#SBATCH --mem=60000
#SBATCH --partition=iric
#SBATCH --mail-type=END
#SBATCH --mail-user=phsun@stanford.edu

python 16.py
