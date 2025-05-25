@echo off
echo [INFO] Running csvModifier.py to filter demonstration data...
python csvModifier.py

echo [INFO] Waiting 5 seconds...
timeout /t 5 >nul

echo [INFO] Running IL-nn.py to train and save bc_model.pth...
python IL-nn.py

echo [INFO] All tasks completed.
pause
