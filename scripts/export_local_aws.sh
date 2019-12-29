#!/bin/bash
# User input
read -p "Instace's Public DNS: " public_dns
read -p "Path to key-pair: " path_key_pair
read -p "Path to store logs unix: " path_logs

# Zip log_all
zip log_all.zip ~/CIFO_project/log_all

# Export data to local machine (run in local machine)
# scp -i "$path_key_pair" ec2-user@"$public_dns":~/CIFO_project/log_all.zip "$path_logs"

# Upload data from local machine (run in local machine)
# scp -i $path_key_pair requirements_aws.txt ec2-user@$public_dns:~/CIFO_project
