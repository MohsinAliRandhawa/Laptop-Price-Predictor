@echo off
title Laptop Price Predictor
color 0A

echo.
echo  ============================================
echo   Laptop Price Predictor - Starting App...
echo  ============================================
echo.
echo  Please wait, opening browser automatically...
echo.

cd /d "%~dp0"

python -m streamlit run "web-app.py" --browser.gatherUsageStats false --browser.serverAddress localhost

pause
