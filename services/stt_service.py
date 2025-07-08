import asyncio
import time
import logging
from typing import Dict, Any

import whisper

logger = logging.getLogger(__name__)

class STTService:
    def __init__(self):
        self.global_whisper = None   # 글로벌 모델 (다국어)
        self.local_whisper = None    # 로컬 모델 (한국어 최적화)
        
    async def initialize(self):
        """모델 초기화"""
        
        logger.info("Whisper 모델들 로드 중...")
        
        try:
            # 1. 글로벌 Whisper (다국어 모델)
            logger.info("📡 글로벌 Whisper 모델 로드 중...")
            self.global_whisper = whisper.load_model("base")  # 다국어 기본 모델
            
            # 2. 로컬 Whisper (한국어 최적화)
            logger.info("🇰🇷 로컬 Whisper 모델 로드 중...")
            # 같은 모델이지만 한국어 최적화 설정으로 사용
            self.local_whisper = whisper.load_model("small")  # 조금 더 큰 모델로 차별화
            
            logger.info("✅ 모든 Whisper 모델 로드 완료!")
            
        except Exception as e:
            logger.error(f"❌ Whisper 모델 로드 실패: {str(e)}")
            raise
    
    async def transcribe_whisper(self, audio_path: str) -> Dict[str, Any]:
        """글로벌 Whisper로 음성 인식 (다국어 모드)"""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._global_whisper_transcribe,
                audio_path
            )
            
            processing_time = time.time() - start_time
            
            return {
                "model": "global_whisper",
                "model_size": "base",
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "confidence": self._calculate_confidence(result),
                "processing_time": round(processing_time, 2),
                "mode": "multilingual",
                "segments_count": len(result.get("segments", []))
            }
            
        except Exception as e:
            logger.error(f"글로벌 Whisper 처리 오류: {str(e)}")
            return {
                "model": "global_whisper",
                "text": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _global_whisper_transcribe(self, audio_path: str):
        """글로벌 Whisper 동기 처리 (언어 자동 감지)"""
        return self.global_whisper.transcribe(
            audio_path,
            # language는 지정하지 않음 (자동 감지)
            fp16=False,
            verbose=False
        )
    
    async def transcribe_korean(self, audio_path: str) -> Dict[str, Any]:
        """로컬 Whisper로 음성 인식 (한국어 최적화)"""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._local_whisper_transcribe,
                audio_path
            )
            
            processing_time = time.time() - start_time
            
            return {
                "model": "local_whisper", 
                "model_size": "small",
                "text": result["text"].strip(),
                "language": "ko",  # 한국어 고정
                "confidence": self._calculate_confidence(result),
                "processing_time": round(processing_time, 2),
                "mode": "korean_optimized",
                "segments_count": len(result.get("segments", []))
            }
            
        except Exception as e:
            logger.error(f"로컬 Whisper 처리 오류: {str(e)}")
            return {
                "model": "local_whisper",
                "text": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _local_whisper_transcribe(self, audio_path: str):
        """로컬 Whisper 동기 처리 (한국어 최적화)"""
        return self.local_whisper.transcribe(
            audio_path,
            language="ko",  # 한국어 강제 지정
            fp16=False,
            verbose=False,
            # 한국어 최적화 옵션들
            initial_prompt="다음은 한국어 음성입니다:",  # 한국어 힌트
            temperature=0.0,  # 더 일관된 결과
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )
    
    def _calculate_confidence(self, result: Dict) -> float:
        """신뢰도 계산"""
        if "segments" in result and result["segments"]:
            # 각 세그먼트의 평균 confidence
            confidences = []
            for segment in result["segments"]:
                if "avg_logprob" in segment:
                    # log probability를 0-1 범위로 변환
                    import math
                    conf = min(1.0, max(0.0, math.exp(segment["avg_logprob"])))
                    confidences.append(conf)
            
            if confidences:
                return round(sum(confidences) / len(confidences), 3)
        
        # 기본값
        return 0.8
    
    async def cleanup(self):
        """리소스 정리"""
        logger.info("STT 서비스 리소스 정리 중...")
        
        if self.global_whisper:
            del self.global_whisper
            self.global_whisper = None
            
        if self.local_whisper:
            del self.local_whisper
            self.local_whisper = None
            
        logger.info("✅ STT 리소스 정리 완료")

# 비교 테스트용 함수
async def compare_whisper_models(audio_path: str):
    """두 Whisper 모델 비교"""
    stt = STTService()
    await stt.initialize()
    
    print(f"🎤 오디오 파일: {audio_path}")
    print("=" * 50)
    
    # 병렬 처리
    tasks = [
        stt.transcribe_whisper(audio_path),
        stt.transcribe_korean(audio_path)
    ]
    
    results = await asyncio.gather(*tasks)
    global_result, local_result = results
    
    print("📡 글로벌 Whisper (다국어):")
    print(f"   텍스트: {global_result.get('text', 'ERROR')}")
    print(f"   언어: {global_result.get('language', 'N/A')}")
    print(f"   신뢰도: {global_result.get('confidence', 0)}")
    print(f"   처리시간: {global_result.get('processing_time', 0)}초")
    
    print("\n🇰🇷 로컬 Whisper (한국어 최적화):")
    print(f"   텍스트: {local_result.get('text', 'ERROR')}")
    print(f"   언어: {local_result.get('language', 'N/A')}")
    print(f"   신뢰도: {local_result.get('confidence', 0)}")
    print(f"   처리시간: {local_result.get('processing_time', 0)}초")
    
    # 간단한 비교
    if len(global_result.get('text', '')) > len(local_result.get('text', '')):
        print("\n🏆 글로벌 모델이 더 많은 텍스트 인식")
    elif len(local_result.get('text', '')) > len(global_result.get('text', '')):
        print("\n🏆 로컬 모델이 더 많은 텍스트 인식")
    else:
        print("\n🤝 비슷한 텍스트 길이")
    
    await stt.cleanup()
    return results

# 테스트용
async def test_stt_service():
    """STT 서비스 테스트"""
    print("STT 서비스 테스트 중...")
    
    stt = STTService()
    await stt.initialize()
    
    print("✅ STT 초기화 성공!")
    print("🎯 오디오 파일을 준비해서 compare_whisper_models() 함수로 테스트하세요!")
    
    await stt.cleanup()

if __name__ == "__main__":
    asyncio.run(test_stt_service())