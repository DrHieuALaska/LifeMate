@echo off
python -B -X utf8 "%~dp0open_review.py" %*
if errorlevel 1 pause
