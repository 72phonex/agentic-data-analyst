@echo off
setlocal
echo ==============================================
echo   Starting Agentic Data Analyst Chatbot...
echo ==============================================

:: Default to small local Qwen model
set OLLAMA_MODEL=qwen2.5:3b

:: Check if Ollama service is reachable on localhost:11434
curl.exe -s --connect-timeout 2 http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Ollama service not detected. Starting in minimized window...
    start /min "Ollama Service" ollama serve
    echo [INFO] Waiting 3 seconds for Ollama to initialize...
    timeout /t 3 /nobreak >nul
) else (
    echo [INFO] Ollama service is already running.
)

:: Run the Streamlit web application
echo [INFO] Launching Streamlit web app...
streamlit run app.py

pause
