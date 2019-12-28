# User input
read -p "Path to project: " proj_path

# Set up conda env in proj_path
cd "$proj_path"
conda create --prefix ./cifo_proj python --no-default-packages
conda activate ./cifo_proj
pip install -r "$proj_path"/requirements.txt
