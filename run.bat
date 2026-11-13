@echo off
chcp 65001 > nul
title Tender AI Lab - Kayan Catering ERP
echo ==========================================================
echo        جاري تشغيل خادم معمل كراسات الشروط (Tender AI Lab)
echo               الرابط: http://127.0.0.1:8055
echo ==========================================================
echo.
echo جاري فتح المتصفح تلقائياً...
start http://127.0.0.1:8055
echo.
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8055 --reload
pause
