# User input
read -p "Instace's Public DNS: " public_dns
read -p "Path to key-pair: " path_key_pair

# Connect to aws instance
chmod 400 "$path_key_pair"
ssh -i "$path_key_pair" ec2-user@"$public_dns"
