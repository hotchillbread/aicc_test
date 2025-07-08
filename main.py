from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import io
import os
from typing import Optional
import speech_recognition as sr
import pyttsx3
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AICC TEST SERVER", version="1.0.0")


# 헬스 체크
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "AI 음성 대화 시스템이 정상 작동중입니다"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)