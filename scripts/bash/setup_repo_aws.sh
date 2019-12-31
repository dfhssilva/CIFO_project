#!/bin/bash
# Clone CIFO_project repository
# git clone https://github.com/DavidSilva98/CIFO_project.git
# cd CIFO_project
# chmod +x ./scripts/*
# Create and set-up environment
python3 -m venv .env
source .env/bin/activate
pip install -r requirements_aws.txt