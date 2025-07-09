import asyncio
import time
import logging
from typing import Dict, Any
import whisper
import torch
from transformers import pipeline
import librosa
import numpy as np

logger = logging.getLogger(__name__)

class STTService:
    def __init__(self):
        self.whisper_model = None
        self.korean_stt_model = None
        self.device = "cpu"  # CPU 사용
        
    async def initialize(self):
        """모델 초기화"""
        logger.info("STT 모델들을 로드하는 중...")
        
        try:
            # Whisper 모델 로드
            logger.info("Whisper 모델 로드 중...")
            self.whisper_model = whisper.load_model("base")  # base 모델 사용 (빠름)
            
            # 한국어 STT 모델 로드 (HuggingFace)
            logger.info("한국어 STT 모델 로드 중...")
            self.korean_stt_model = pipeline(
                "automatic-speech-recognition",
                model="kresnik/wav2vec2-large-xlsr-korean",
                device=-1,  # CPU 사용
                torch_dtype=torch.float32
            )
            
            logger.info("✅ STT 모델 로드 완료!")
            
        except Exception as e:
            logger.error(f"❌ STT 모델 로드 실패: {str(e)}")
            raise
    
    async def transcribe_whisper(self, audio_path: str) -> Dict[str, Any]:
        """Whisper로 음성 인식"""
        start_time = time.time()
        
        try:
            # Whisper는 동기 함수이므로 thread에서 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._whisper_transcribe_sync,
                audio_path
            )
            
            processing_time = time.time() - start_time
            
            return {
                "model": "whisper",
                "text": result["text"].strip(),
                "language": result.get("language", "ko"),
                "confidence": self._calculate_whisper_confidence(result),
                "processing_time": round(processing_time, 2),
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            logger.error(f"Whisper 처리 오류: {str(e)}")
            return {
                "model": "whisper",
                "text": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _whisper_transcribe_sync(self, audio_path: str):
        """Whisper 동기 처리"""
        return self.whisper_model.transcribe(
            audio_path,
            language="ko",  # 한국어로 고정
            fp16=False  # CPU에서는 fp16 사용 안함
        )
    
    def _calculate_whisper_confidence(self, result: Dict) -> float:
        """Whisper 신뢰도 계산"""
        if "segments" in result and result["segments"]:
            # 각 세그먼트의 평균 confidence
            confidences = []
            for segment in result["segments"]:
                if "avg_logprob" in segment:
                    # log probability를 confidence로 변환
                    conf = min(1.0, max(0.0, np.exp(segment["avg_logprob"])))
                    confidences.append(conf)
            
            if confidences:
                return round(np.mean(confidences), 3)
        
        # 기본값
        return 0.8
    
    async def transcribe_korean(self, audio_path: str) -> Dict[str, Any]:
        """한국어 특화 STT로 음성 인식"""
        start_time = time.time()
        
        try:
            # 오디오 전처리
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            
            # HuggingFace pipeline은 동기 함수이므로 thread에서 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._korean_stt_transcribe_sync,
                audio_array
            )
            
            processing_time = time.time() - start_time
            
            return {
                "model": "korean_stt",
                "text": result["text"].strip(),
                "confidence": result.get("score", 0.85),
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            logger.error(f"한국어 STT 처리 오류: {str(e)}")
            return {
                "model": "korean_stt", 
                "text": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _korean_stt_transcribe_sync(self, audio_array: np.ndarray):
        """한국어 STT 동기 처리"""
        return self.korean_stt_model(audio_array)
    
    async def cleanup(self):
        """리소스 정리"""
        logger.info("STT 서비스 리소스 정리 중...")
        
        if self.whisper_model:
            del self.whisper_model
            self.whisper_model = None
            
        if self.korean_stt_model:
            del self.korean_stt_model
            self.korean_stt_model = None
            
        # GPU 메모리 정리 (사용하는 경우)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("✅ STT 리소스 정리 완료")

# 테스트용 함수들
async def test_stt_service():
    """STT 서비스 테스트"""
    stt = STTService()
    await stt.initialize()
    
    result1 = await stt.transcribe_whisper("test.wav")
    result2 = await stt.transcribe_korean("test.wav")
    print("Whisper:", result1)
    print("Korean STT:", result2)
    
    await stt.cleanup()

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_stt_service())