#!/bin/bash

# Activate the virtual environment
source ./venv/bin/activate

# Pass all arguments to launch.py
python launch.py "$@"