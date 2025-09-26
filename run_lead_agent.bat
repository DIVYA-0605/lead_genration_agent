@echo off
cd /d "%~dp0"
call env\Scripts\activate
streamlit run ai_lead_generation_agent_ollama.py --server.address 0.0.0.0 --server.port 8501
pause