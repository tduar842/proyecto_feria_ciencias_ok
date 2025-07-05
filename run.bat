@echo off
set programa_corriendo=4

IF %programa_corriendo%==1 (start "" "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\venv\Scripts\python.exe" "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\object_detector.py")
IF %programa_corriendo%==2 (start "" "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\venv\Scripts\python.exe" "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\object_detector_video.py")
IF %programa_corriendo%==3 (start "" "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\venv\Scripts\python.exe" "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\face_detector.py")
IF %programa_corriendo%==4 (start "" "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\venv\Scripts\python.exe" "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\face_detector_video.py")

