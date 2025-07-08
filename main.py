from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import time
import base64
import io
import tempfile
import os
from typing import Dict, Any
import logging

# 모델 imports (나중에 구현할 서비스들)
from services.stt_service import STTService
from services.llm_service import LLMService  
from services.tts_service import TTSService
from utils.audio_utils import AudioProcessor
from utils.evaluation import AccuracyEvaluator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="AICC API Server",
    description="AI Call Center with STT/LLM/TTS comparison",
    version="1.0.0"
)

#cors 설정 (프론트 안붙일거라 필요는 없음)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#서비스 인스턴스 (서버 시작 시 초기화)
stt_service = None
llm_service = None
tts_service = None
audio_processor = None
evaluator = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모든 모델 로드"""
    global stt_service, llm_service, tts_service, audio_processor, evaluator
    
    logger.info("서버 시작 중... 모델들을 로드합니다.")
    
    try:
        # STT 서비스 초기화
        logger.info("STT 모델 로드 중...")
        stt_service = STTService()
        await stt_service.initialize()
        
        # LLM 서비스 초기화
        logger.info("LLM 모델 로드 중...")
        llm_service = LLMService()
        await llm_service.initialize()
        
        # TTS 서비스 초기화
        logger.info("TTS 모델 로드 중...")
        tts_service = TTSService()
        await tts_service.initialize()
        
        # 유틸리티 초기화
        audio_processor = AudioProcessor()
        evaluator = AccuracyEvaluator()
        
        logger.info("모든 모델 로드 완료!")
        
    except Exception as e:
        logger.error(f"모델 로드 실패: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 리소스 정리"""
    logger.info("서버 종료 중...")
    
    if stt_service:
        await stt_service.cleanup()
    if llm_service:
        await llm_service.cleanup()
    if tts_service:
        await tts_service.cleanup()
        
    logger.info("리소스 정리 완료!")

@app.get("/")
async def root():
    """서버 상태 체크"""
    return {
        "message": "AICC API Server is running!",
        "status": "healthy",
        "models": {
            "stt_loaded": stt_service is not None,
            "llm_loaded": llm_service is not None,
            "tts_loaded": tts_service is not None
        }
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "stt": "ready" if stt_service else "not_ready",
            "llm": "ready" if llm_service else "not_ready",
            "tts": "ready" if tts_service else "not_ready"
        }
    }

@app.post("/api/v1/process-audio")
async def process_audio_pipeline(file: UploadFile = File(...)):
    """
    통합 오디오 처리 파이프라인
    STT → LLM → TTS 전체 과정을 한 번에 처리
    """
    if not all([stt_service, llm_service, tts_service]):
        raise HTTPException(status_code=503, detail="서비스가 아직 준비되지 않았습니다.")
    
    start_time = time.time()
    
    try:
        # 1. 오디오 파일 검증 및 전처리
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="오디오 파일만 업로드 가능합니다.")
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # 오디오 전처리
        processed_audio = audio_processor.preprocess(tmp_file_path)
        
        # 2. STT 처리 (병렬)
        logger.info("STT 처리 시작...")
        stt_start = time.time()
        
        stt_tasks = [
            stt_service.transcribe_whisper(processed_audio),
            stt_service.transcribe_korean(processed_audio)
        ]
        stt_results = await asyncio.gather(*stt_tasks)
        
        # STT 정확도 비교
        stt_comparison = evaluator.compare_stt_results(stt_results)
        best_stt_text = stt_comparison['best_result']['text']
        
        stt_time = time.time() - stt_start
        logger.info(f"STT 처리 완료 ({stt_time:.2f}s)")
        
        # 3. LLM 처리 (병렬)
        logger.info("LLM 처리 시작...")
        llm_start = time.time()
        
        llm_tasks = [
            llm_service.generate_openai(best_stt_text),
            llm_service.generate_korean(best_stt_text)
        ]
        llm_results = await asyncio.gather(*llm_tasks)
        
        # LLM 정확도 비교
        llm_comparison = evaluator.compare_llm_results(llm_results, best_stt_text)
        best_llm_text = llm_comparison['best_result']['text']
        
        llm_time = time.time() - llm_start
        logger.info(f"LLM 처리 완료 ({llm_time:.2f}s)")
        
        # 4. TTS 처리
        logger.info("TTS 처리 시작...")
        tts_start = time.time()
        
        tts_audio = await tts_service.generate_speech(best_llm_text)
        tts_audio_base64 = base64.b64encode(tts_audio).decode('utf-8')
        
        tts_time = time.time() - tts_start
        logger.info(f"TTS 처리 완료 ({tts_time:.2f}s)")
        
        # 5. 결과 반환
        total_time = time.time() - start_time
        
        result = {
            "success": True,
            "stt_results": {
                "whisper": stt_results[0],
                "korean_stt": stt_results[1],
                "comparison": stt_comparison,
                "selected_text": best_stt_text
            },
            "llm_results": {
                "openai": llm_results[0],
                "korean_llm": llm_results[1],
                "comparison": llm_comparison,
                "selected_text": best_llm_text
            },
            "tts_result": {
                "audio_base64": tts_audio_base64,
                "format": "wav"
            },
            "processing_time": {
                "stt": round(stt_time, 2),
                "llm": round(llm_time, 2),
                "tts": round(tts_time, 2),
                "total": round(total_time, 2)
            },
            "metadata": {
                "original_filename": file.filename,
                "file_size": len(content),
                "timestamp": time.time()
            }
        }
        
        # 임시 파일 삭제
        os.unlink(tmp_file_path)
        
        logger.info(f"전체 처리 완료! 총 {total_time:.2f}초")
        return result
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        # 임시 파일 삭제
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/v1/stt/compare")
async def compare_stt_only(file: UploadFile = File(...)):
    """STT만 비교하는 엔드포인트"""
    if not stt_service:
        raise HTTPException(status_code=503, detail="STT 서비스가 준비되지 않았습니다.")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        processed_audio = audio_processor.preprocess(tmp_file_path)
        
        # 병렬 STT 처리
        stt_tasks = [
            stt_service.transcribe_whisper(processed_audio),
            stt_service.transcribe_korean(processed_audio)
        ]
        stt_results = await asyncio.gather(*stt_tasks)
        
        # 결과 비교
        comparison = evaluator.compare_stt_results(stt_results)
        
        os.unlink(tmp_file_path)
        
        return {
            "whisper_result": stt_results[0],
            "korean_stt_result": stt_results[1],
            "comparison": comparison
        }
        
    except Exception as e:
        logger.error(f"STT 비교 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/llm/compare")
async def compare_llm_only(text: str):
    """LLM만 비교하는 엔드포인트"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM 서비스가 준비되지 않았습니다.")
    
    try:
        # 병렬 LLM 처리
        llm_tasks = [
            llm_service.generate_openai(text),
            llm_service.generate_korean(text)
        ]
        llm_results = await asyncio.gather(*llm_tasks)
        
        # 결과 비교
        comparison = evaluator.compare_llm_results(llm_results, text)
        
        return {
            "openai_result": llm_results[0],
            "korean_llm_result": llm_results[1],
            "comparison": comparison
        }
        
    except Exception as e:
        logger.error(f"LLM 비교 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/tts/generate")
async def generate_tts_only(text: str):
    """TTS만 생성하는 엔드포인트"""
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS 서비스가 준비되지 않았습니다.")
    
    try:
        tts_audio = await tts_service.generate_speech(text)
        tts_audio_base64 = base64.b64encode(tts_audio).decode('utf-8')
        
        return {
            "audio_base64": tts_audio_base64,
            "format": "wav",
            "text": text
        }
        
    except Exception as e:
        logger.error(f"TTS 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 시에만 사용
        log_level="info"
    )