#!/bin/bash

# Enable automate interrupt in case of errors
set -e

# Variables
BUILD_DIR="../build"
VIDEO_DIR="$BUILD_DIR/video"
TEST_DIR="$BUILD_DIR/src"
EXECUTABLE_NAME="main"
EXECUTABLE_PATH="$TEST_DIR/$EXECUTABLE_NAME"

# Vérification des arguments
if [ "$#" -ne 3 ]; then
    echo "Usage : $0 <nbsec> <fps> <nbtours>"
    exit 1
fi

NB_SEC=$1
FPS=$2
NB_TOUR=$3

# Vérifie si les valeurs sont valides
if [ "$NB_SEC" -gt 300 ] || [ "$FPS" -le 0 ] || [ "$NB_TOUR" -le 0 ]; then
    echo "Erreur : paramètres invalides."
    echo "- Le nombre de secondes ne doit pas dépasser 300"
    echo "- Le nombre de fps doit être > 0"
    echo "- Le nombre de tours doit être > 0"
    exit 2
fi

# Création des répertoires nécessaires
[ ! -d "$VIDEO_DIR" ] && echo "Creating video dir..." && mkdir -p "$VIDEO_DIR"

# Verify build/ existency and create it if necessary
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found! Creating it..."
    mkdir -p "$BUILD_DIR"
fi

# Verify build/test/ existency and create it if necessary
if [ ! -d "$TEST_DIR" ]; then
    echo "Test directory not found! Creating it..."
    mkdir -p "$TEST_DIR"
fi

# Go in build/ folder
cd "$BUILD_DIR"

# Step 1: Run cmake
echo "Running cmake..."
cmake ..
if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Step 2: Run make
echo "Running make..."
make
if [ $? -ne 0 ]; then
    echo "Make process failed!"
    exit 1
fi

# Step 3: Run the executable
echo "Running the executable..."
if [ -f "$EXECUTABLE_PATH" ]; then
    "$EXECUTABLE_PATH" "$NB_SEC" "$FPS" "$NB_TOUR"
else
    echo "Executable $EXECUTABLE_NAME not found in $TEST_DIR!"
    exit 1
fi

echo "Script finished successfully!"
