#!/bin/bash

#SBATCH -A NAISS2025-5-98 -p alvis
#SBATCH --job-name=aco_active_learning
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=1
#SBATCH -t 0-75:00:0
#SBATCH --gpus-per-node=A40:1

# Carica moduli se necessario (decommenta se serve)
module load Python/3.11.3-GCCcore-12.3.0

# Attiva l'ambiente virtuale
source ~/venvs/alessia-ml-env/bin/activate

# Esegui lo script Python desiderato
#python C_r.py
#python alt3.py
#python toy.py
#python main.py
#python surr_model_toy.py
python Crif_project.py 




