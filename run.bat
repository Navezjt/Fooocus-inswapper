@echo off
setlocal

rem Activate the virtual environment
call .\venv\Scripts\activate

rem Pass all arguments to launch.py
python launch.py %*