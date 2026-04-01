@echo off
cd /d "C:\NEW SAR"
echo Starting Image Colorization App...
streamlit run app.py --server.headless=true --server.port=8501
pause
