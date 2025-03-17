@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM Set path to 7z
SET SEVENZIP="C:\Program Files\7-Zip\7z.exe"

IF NOT EXIST %1 (
    echo Use: unpack_midair.bat "path_to_MidAir_dataset_folder"
    exit /b 1
)

FOR /R "%~1" %%I IN (*.zip) DO (
    echo Unpack the archive: %%I
    %SEVENZIP% x "%%I" -o"%%~dpI" -y >nul
	DEL "%%I"
    IF %ERRORLEVEL% NEQ 0 (
        echo Error unpacking file %%I
    ) ELSE (
        echo Unpack successful: %%I
    )
)

echo All archives unpacked!
ENDLOCAL
pause
