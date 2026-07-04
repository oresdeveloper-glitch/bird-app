@echo off
REM Automated GitHub Setup Script for Bird App
REM This script initializes Git, adds files, and prepares for GitHub push

echo.
echo ========================================
echo Bird App - GitHub Setup Automation
echo ========================================
echo.

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed!
    echo.
    echo Download Git from: https://git-scm.com/download/win
    echo Then run this script again.
    pause
    exit /b 1
)

echo [1/5] Initializing Git repository...
git init
if errorlevel 1 goto error

echo [2/5] Configuring Git user...
git config user.name "Bird App Developer"
git config user.email "dev@birdapp.local"
if errorlevel 1 goto error

echo [3/5] Adding all files (respecting .gitignore)...
git add .
if errorlevel 1 goto error

echo [4/5] Creating initial commit...
git commit -m "Initial commit: Bird species identification web app"
if errorlevel 1 goto error

echo.
echo ========================================
echo SUCCESS! Local Git repository ready
echo ========================================
echo.
echo Next steps:
echo 1. Go to https://github.com/new
echo 2. Create a new repository (same name as this folder)
echo 3. Run these commands:
echo.
echo    git branch -M main
echo    git remote add origin https://github.com/YOUR-USERNAME/bird-app.git
echo    git push -u origin main
echo.
echo Replace YOUR-USERNAME with your GitHub username
echo.
pause
goto end

:error
echo.
echo ERROR occurred during setup!
echo.
pause
exit /b 1

:end
