@echo off
echo Deleting bc_model.pth if it exists...
del /f /q bc_model.pth 2>nul

echo Copying bc_model_P2.pth to bc_model.pth...
copy /y bc_model_J3.pth bc_model.pth

echo Starting SERVER...
start "SERVER" cmd /k python tmrl\__main__.py --server

echo Waiting for server port 55555 to be open...
:waitloop
powershell -Command "exit !(Test-NetConnection 127.0.0.1 -Port 55555).TcpTestSucceeded"
if errorlevel 1 (
    timeout /t 1 >nul
    goto waitloop
)

echo Starting TRAINER...
start "TRAINER" cmd /k python tmrl\__main__.py --trainer

echo Waiting 5 seconds...
timeout /t 5 /nobreak >nul

echo Starting WORKER...
start "WORKER" cmd /k python tmrl\__main__.py --worker

echo All components launched.