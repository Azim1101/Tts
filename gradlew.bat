@echo off
rem Thin Gradle launcher for Swara (see gradlew for details).
setlocal

if defined GRADLE_HOME (
  if exist "%GRADLE_HOME%\bin\gradle.bat" (
    call "%GRADLE_HOME%\bin\gradle.bat" %*
    exit /b %ERRORLEVEL%
  )
)
where gradle >nul 2>nul
if %ERRORLEVEL%==0 (
  gradle %*
  exit /b %ERRORLEVEL%
)

echo ERROR: Gradle 8.7 not found. Install a Gradle 8.x or set GRADLE_HOME.
exit /b 1
