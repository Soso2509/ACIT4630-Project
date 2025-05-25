@echo off
echo Running: python setup.py install
python setup.py --install

echo Waiting 5 seconds...
timeout /t 5 /nobreak >nul

echo Running: python setup.py build
python setup.py --build

echo Waiting 5 seconds...
timeout /t 5 /nobreak >nul

echo Running: python tmrl/__main__.py --install
python tmrl/__main__.py --install

echo All steps completed.
pause
