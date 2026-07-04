"""
Automated GitHub Setup Script for Bird App
Run this if batch/shell scripts don't work on your system
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"[*] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"    ✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
        return False

def main():
    print("\n" + "="*50)
    print("Bird App - GitHub Setup Automation (Python)")
    print("="*50 + "\n")
    
    # Check if Git is installed
    if not run_command("git --version", "Checking Git installation"):
        print("\n❌ ERROR: Git is not installed!")
        print("Download from: https://git-scm.com/download/win")
        sys.exit(1)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    print(f"\n📁 Working directory: {project_dir}\n")
    
    # Run setup steps
    steps = [
        ("git init", "Initializing Git repository"),
        ('git config user.name "Bird App Developer"', "Setting Git user name"),
        ('git config user.email "dev@birdapp.local"', "Setting Git email"),
        ("git add .", "Adding all files"),
        ('git commit -m "Initial commit: Bird species identification web app"', "Creating initial commit"),
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"\n❌ Setup failed at: {desc}")
            sys.exit(1)
    
    # Success message
    print("\n" + "="*50)
    print("✅ SUCCESS! Local Git repository ready")
    print("="*50)
    print("\n📋 Next steps:")
    print("1. Go to https://github.com/new")
    print("2. Create a new repository")
    print("3. Run these commands in your terminal:\n")
    print("   git branch -M main")
    print("   git remote add origin https://github.com/YOUR-USERNAME/bird-app.git")
    print("   git push -u origin main\n")
    print("💡 Replace YOUR-USERNAME with your GitHub username\n")

if __name__ == "__main__":
    main()
