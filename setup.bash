## Setup script for project to set python virtual environment

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "----------------------------------------"
echo "Virtual environment setup complete. To activate, run 'source .venv/bin/activate' It is recommended to add this line to your .bashrc or .zshrc file for automatic activation."
