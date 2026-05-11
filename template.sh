# Creating directories
mkdir -p src
mkdir -p research

# Creating files
touch src/__init__.py
touch src/helper.py
touch src/prompt.py
touch .env
touch setup.py
touch app.py
touch research/trials.ipynb
touch requirements.txt


echo "Project structure created successfully!"

# Virtual environment setup
conda create -n myenv python=3.10 -y
conda activate myenv

# Installing dependencies
pip install -r requirements.txt
