@echo off
echo Starting SERVER...
start "SERVER" cmd /k python tmrl\__main__.py --server

echo Waiting for server port 55555 to be open...
:waitloop
powershell -Command "exit !(Test-NetConnection 127.0.0.1 -Port 55555).TcpTestSucceeded"
if errorlevel 1 (
    timeout /t 1 >nul
    goto waitloop
)


echo Starting WORKER...
start "IMITATION WORKER" cmd /k python tmrl\__main__.py --imitation-worker

echo All components launched.
