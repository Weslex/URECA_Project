#!/bin/bash

module load singularity

singularity exec --nv /home/jphillips/images/csci4850-2023-Spring.sif ./PretrainingViT.py
