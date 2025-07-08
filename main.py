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

app = FastAPI(title="AI 음성 대화 시스템", version="1.0.0")

# 환경 변수 설정 (실제 사용시 .env 파일이나 환경변수로 설정)
openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# 모델 설정
class ConversationRequest(BaseModel):
    text: str
    voice_type: Optional[str] = "korean"
    speed: Optional[int] = 150

class ConversationResponse(BaseModel):
    original_text: str
    response_text: str
    audio_generated: bool

# STT 초기화
recognizer = sr.Recognizer()

# TTS 초기화
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)

# 스레드 풀 실행자
executor = ThreadPoolExecutor(max_workers=4)

@app.get("/")
def start_test():
    return {
        "message": "AI 음성 대화 시스템이 실행중입니다",
        "endpoints": {
            "stt": "/stt - 음성을 텍스트로 변환",
            "llm": "/llm - 텍스트 기반 대화",
            "tts": "/tts - 텍스트를 음성으로 변환",
            "conversation": "/conversation - 전체 대화 파이프라인"
        }
    }

@app.post("/stt")
async def speech_to_text(audio_file: UploadFile = File(...)):
    """음성 파일을 텍스트로 변환"""
    try:
        # 업로드된 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # 음성 인식 수행
        def recognize_speech():
            try:
                with sr.AudioFile(tmp_file_path) as source:
                    # 노이즈 제거
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.record(source)
                
                # Google STT 사용 (한국어 지원)
                text = recognizer.recognize_google(audio, language='ko-KR')
                return text
            except sr.UnknownValueError:
                raise HTTPException(status_code=400, detail="음성을 인식할 수 없습니다")
            except sr.RequestError as e:
                raise HTTPException(status_code=500, detail=f"음성 인식 서비스 오류: {e}")
            finally:
                # 임시 파일 삭제
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        # 비동기로 음성 인식 실행
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(executor, recognize_speech)
        
        return {
            "recognized_text": text,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"STT 오류: {e}")
        raise HTTPException(status_code=500, detail=f"음성 변환 중 오류가 발생했습니다: {str(e)}")

@app.post("/llm")
async def llm_response(request: ConversationRequest):
    """텍스트를 입력받아 AI 응답 생성"""
    try:
        # OpenAI GPT를 사용한 응답 생성
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 친근하고 도움이 되는 한국어 AI 어시스턴트입니다. 자연스럽고 따뜻한 대화를 해주세요."},
                {"role": "user", "content": request.text}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        return {
            "user_input": request.text,
            "ai_response": ai_response,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"LLM 오류: {e}")
        raise HTTPException(status_code=500, detail=f"AI 응답 생성 중 오류가 발생했습니다: {str(e)}")

@app.post("/tts")
async def text_to_speech(request: ConversationRequest):
    """텍스트를 음성으로 변환"""
    try:
        def generate_speech():
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file_path = tmp_file.name
            
            # TTS 설정
            tts_engine.setProperty('rate', request.speed)
            
            # 한국어 음성 설정 (시스템에 한국어 TTS가 있는 경우)
            voices = tts_engine.getProperty('voices')
            for voice in voices:
                if 'korean' in voice.name.lower() or 'ko' in voice.id.lower():
                    tts_engine.setProperty('voice', voice.id)
                    break
            
            # 음성 파일 생성
            tts_engine.save_to_file(request.text, tmp_file_path)
            tts_engine.runAndWait()
            
            # 파일 읽기
            with open(tmp_file_path, 'rb') as f:
                audio_data = f.read()
            
            # 임시 파일 삭제
            os.unlink(tmp_file_path)
            
            return audio_data
        
        # 비동기로 음성 생성 실행
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(executor, generate_speech)
        
        # 음성 파일을 스트리밍으로 반환
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=response.wav"}
        )
    
    except Exception as e:
        logger.error(f"TTS 오류: {e}")
        raise HTTPException(status_code=500, detail=f"음성 생성 중 오류가 발생했습니다: {str(e)}")

@app.post("/conversation")
async def full_conversation(audio_file: UploadFile = File(...)):
    """전체 대화 파이프라인: STT → LLM → TTS"""
    try:
        # 1. STT: 음성을 텍스트로 변환
        logger.info("STT 처리 시작")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        def recognize_speech():
            try:
                with sr.AudioFile(tmp_file_path) as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.record(source)
                text = recognizer.recognize_google(audio, language='ko-KR')
                return text
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        loop = asyncio.get_event_loop()
        user_text = await loop.run_in_executor(executor, recognize_speech)
        logger.info(f"STT 결과: {user_text}")
        
        # 2. LLM: AI 응답 생성
        logger.info("LLM 처리 시작")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 친근하고 도움이 되는 한국어 AI 어시스턴트입니다. 자연스럽고 따뜻한 대화를 해주세요."},
                {"role": "user", "content": user_text}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content.strip()
        logger.info(f"LLM 결과: {ai_response}")
        
        # 3. TTS: 텍스트를 음성으로 변환
        logger.info("TTS 처리 시작")
        def generate_speech():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file_path = tmp_file.name
            
            tts_engine.save_to_file(ai_response, tmp_file_path)
            tts_engine.runAndWait()
            
            with open(tmp_file_path, 'rb') as f:
                audio_data = f.read()
            
            os.unlink(tmp_file_path)
            return audio_data
        
        audio_data = await loop.run_in_executor(executor, generate_speech)
        
        # 결과 반환 (음성 파일과 텍스트 정보)
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=ai_response.wav",
                "X-User-Text": user_text,
                "X-AI-Response": ai_response
            }
        )
    
    except Exception as e:
        logger.error(f"대화 파이프라인 오류: {e}")
        raise HTTPException(status_code=500, detail=f"대화 처리 중 오류가 발생했습니다: {str(e)}")

# 헬스 체크
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "AI 음성 대화 시스템이 정상 작동중입니다"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)