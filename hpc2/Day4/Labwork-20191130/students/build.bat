@echo off
set VCINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio 15.0\VC
setlocal
call :initColorPrint
if not exist build (
    call :colorPrint 4e "Creates the build directory" /n
    mkdir build
)
call :colorPrint 4e "Move into ./build" /n
cd build
call :colorPrint 4e "Configure the projet ..." /n
cmake -DCUDA_SAMPLES_INC="C:/ProgramData/NVIDIA Corporation/Cuda Samples/v10.0/common/inc" -G"Visual Studio 15 2017 Win64" ..

call :colorPrint 4e "Build the project ..." /n
for %%I IN (exo1,exo2,exo3,exo4) DO (
    call :buildTarget %%I
)

REM bye bye
call :colorPrint 4e "That's all, folks!" /n
call :cleanupColorPrint
exit /b


REM This function build one target
:buildTarget target
call :colorPrint 4e "Build TARGET %1:" /n
cmake --build . --config Release --target %1 -- /verbosity:quiet
exit /b

REM This function print a string, and may be a newline
:colorPrint Color  Str  [/n]
setlocal
set "str=%~2"
call :colorPrintVar %1 str %3
exit /b

REM this function is used to print a string
:colorPrintVar  Color  StrVar  [/n]
if not defined %~2 exit /b
setlocal enableDelayedExpansion
set "str=a%DEL%!%~2:\=a%DEL%\..\%DEL%%DEL%%DEL%!"
set "str=!str:/=a%DEL%/..\%DEL%%DEL%%DEL%!"
set "str=!str:"=\"!"
pushd "%temp%"
findstr /p /A:%1 "." "!str!\..\x" nul
if /i "%~3"=="/n" echo(
exit /b

REM this function initializes the color print system
:initColorPrint
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do set "DEL=%%a"
<nul >"%temp%\x" set /p "=%DEL%%DEL%%DEL%%DEL%%DEL%%DEL%.%DEL%"
exit /b

REM this function ends the color print system
:cleanupColorPrint
del "%temp%\x"
exit /b
