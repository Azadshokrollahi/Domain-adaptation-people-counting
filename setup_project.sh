#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/public_data
mkdir -p data/our_dataset/room_a
mkdir -p data/our_dataset/room_b
mkdir -p data/our_dataset/room_c

mkdir -p src
mkdir -p configs
mkdir -p scripts
mkdir -p results

touch README.md
touch requirements.txt
touch Dockerfile
touch docker-compose.yml
touch Makefile
touch .gitignore
touch .dockerignore

touch data/README.md
touch results/.gitkeep

# Python files
touch src/config.py
touch src/preprocess_public_data.py
touch src/main_regression_pretrain.py
touch src/feature_based_methods.py
touch src/parameter_based_methods.py

# Config files
touch configs/rooms.yaml

echo "Project structure created."
