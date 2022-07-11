#!/bin/bash
#SBATCH -J segmentCells                   		# Job name
#SBATCH -o segmentCells_%j.out        		# Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100   	      			# Queue name (you use the development queue to test jobs that take less than 2 hours)
#SBATCH -N 1                  			# Total number of nodes requested
#SBATCH -n 1                 			# Total number of threas tasks requested (128 per node)
#SBATCH -t 01:00:00           			# Run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=tyler_jost@utexas.edu   	# Address email notifications
#SBATCH --mail-type=all				# Email at begin and end of job
#SBATCH -A DMS21043      				# <-- Allocation name to charge job against

conda activate computerVision
cd /scratch/08002/tjost/cellMorph
echo "Starting analysis"
python segmentCells.py
