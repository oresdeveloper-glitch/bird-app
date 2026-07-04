#!/bin/bash
# Automated GitHub Setup Script for Bird App
# For macOS/Linux

set -e  # Exit on error

echo ""
echo "========================================"
echo "Bird App - GitHub Setup Automation"
echo "========================================"
echo ""

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed!"
    echo ""
    echo "Install Git using:"
    echo "  macOS: brew install git"
    echo "  Linux: sudo apt-get install git"
    echo ""
    exit 1
fi

echo "[1/5] Initializing Git repository..."
git init

echo "[2/5] Configuring Git user..."
git config user.name "Bird App Developer"
git config user.email "dev@birdapp.local"

echo "[3/5] Adding all files (respecting .gitignore)..."
git add .

echo "[4/5] Creating initial commit..."
git commit -m "Initial commit: Bird species identification web app"

echo ""
echo "========================================"
echo "SUCCESS! Local Git repository ready"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Go to https://github.com/new"
echo "2. Create a new repository (same name as this folder)"
echo "3. Run these commands:"
echo ""
echo "   git branch -M main"
echo "   git remote add origin https://github.com/YOUR-USERNAME/bird-app.git"
echo "   git push -u origin main"
echo ""
echo "Replace YOUR-USERNAME with your GitHub username"
echo ""
