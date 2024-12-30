#!/bin/bash

set -xe

# Define variables
PROJECT_DIR="my_mega_model"
REQUIREMENTS_FILE="requirements.txt"
PYTHON_SCRIPT="uniform_coloring.py"

# Create a project folder
mkdir -p $PROJECT_DIR

# Navigate to the project folder
cd $PROJECT_DIR

# Create a virtual environment
/usr/bin/python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Copy requirements.txt from the venv folder
cp ../$REQUIREMENTS_FILE .

# Copy the Python script from the venv folder
cp ../$PYTHON_SCRIPT .

# Upgrade pip
python3 -m pip install -U pip

# Install the requirements
pip install -r $REQUIREMENTS_FILE

mkdir -p out

# Run the Python script
python3 $PYTHON_SCRIPT

# Deactivate the virtual environment
deactivate

mv ./out.zip ../

# Navigate back to the original directory
cd ../

# Remove the project folder and its contents
rm -rf $PROJECT_DIR

# Print completion message
echo "Script executed successfully. Virtual environment deactivated, and project directory removed."
