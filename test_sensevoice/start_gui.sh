#!/bin/bash
cd "$(dirname "$0")"
source ~/.zshrc
conda activate sensevoice
python run_m1_sensevoice_gui.py


