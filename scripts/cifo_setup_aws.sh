#!/usr/bin/bash
# User input
read -p "Instace's Public DNS: " public_dns
read -p "Path to key-pair: " path_key_pair
read -p "Path to store logs: " path_logs

# Update packages
sudo yum update # sudo apt-get update for ubuntu instance
sudo yum install python3
sudo yum install git

# Clone CIFO_project repository
git clone https://github.com/DavidSilva98/CIFO_project.git
cd CIFO_project

# Create and set-up environment
python3 -m venv .env
source .env/bin/activate
pip install -r requirements_aws.txt

# Run & Create logs
python ga_script_TSP_data.py

# Zip log_all
zip log_all.zip ~/CIFO_project/log_all

# Export data to local machine
scp -i "$path_key_pair" ec2-user@"$public_dns":~/CIFO_project/log_all.zip "$path_logs"

# Upload data from local machine
# scp -i $path_key_pair requirements_aws.txt ec2-user@$public_dns:~/CIFO_project
