@echo off
set programa_corriendo=%1
set path_python="C:\Users\PC GAMER 2025\Downloads\proyecto_feria\venv\Scripts\python.exe"

IF %programa_corriendo%==1 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_1\object_detector.py")
IF %programa_corriendo%==2 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_1\object_detector_video.py")
IF %programa_corriendo%==3 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_2\face_detector.py")
IF %programa_corriendo%==4 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_2\face_detector_video.py")
IF %programa_corriendo%==5 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_3\extract_faces.py")
IF %programa_corriendo%==6 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_3\f_detector_video_cap.py")
IF %programa_corriendo%==7 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_4\face_detector_videocap.py")
IF %programa_corriendo%==8 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_5\captura_rostro.py")
IF %programa_corriendo%==9 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_5\entrenar_rf.py")
IF %programa_corriendo%==10 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_5\reconocimiento_facial.py")
IF %programa_corriendo%==11 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_6\figuras_geometricas_deteccion.py")
IF %programa_corriendo%==12 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\capitulo_7\figuras_geometricas_colores_deteccion.py")
IF %programa_corriendo%==13 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\py_vision\main.py")
IF %programa_corriendo%==14 (start "" %path_python% "C:\Users\PC GAMER 2025\Downloads\proyecto_feria\py_vision\rrt.py")
