import asyncio
import time
import logging
import io
from typing import Dict, Any

# Google TTS만 사용
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    print("⚠️ gtts가 설치되지 않음")
    GTTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.is_initialized = False
        
    async def initialize(self):
        """TTS 서비스 초기화"""
        logger.info("Google TTS 서비스 초기화 중...")
        
        if not GTTS_AVAILABLE:
            logger.error("❌ Google TTS가 설치되지 않았습니다.")
            logger.info("pip install gtts 명령어로 설치하세요")
            raise ImportError("gtts가 설치되지 않음")
        
        # Google TTS는 별도 초기화 불필요 (온라인 서비스)
        self.is_initialized = True
        logger.info("✅ Google TTS 서비스 초기화 완료!")
    
    async def generate_speech(self, text: str) -> bytes:
        """텍스트를 음성으로 변환 (Google TTS)"""
        if not self.is_initialized:
            raise RuntimeError("TTS 서비스가 초기화되지 않음")
        
        start_time = time.time()
        logger.info(f"🔊 TTS 생성 시작: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # 비동기로 Google TTS 처리
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None,
                self._generate_gtts_sync,
                text
            )
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Google TTS 생성 완료 ({processing_time:.2f}초, {len(audio_data)} bytes)")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Google TTS 생성 오류: {str(e)}")
            raise
    
    def _generate_gtts_sync(self, text: str) -> bytes:
        """Google TTS 동기 처리"""
        # gTTS 객체 생성
        tts = gTTS(
            text=text,
            lang='ko',      # 한국어
            slow=False,     # 일반 속도
            tld='com'       # Google 도메인 (.com)
        )
        
        # 메모리 버퍼에 저장
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.read()
    
    async def cleanup(self):
        """리소스 정리"""
        logger.info("TTS 서비스 리소스 정리 중...")
        self.is_initialized = False
        logger.info("✅ TTS 리소스 정리 완료")

# 테스트용 함수
async def test_tts_service():
    """TTS 서비스 테스트"""
    print("🔊 Google TTS 서비스 테스트 중...")
    
    tts = TTSService()
    await tts.initialize()
    
    # 테스트 텍스트들
    test_texts = [
        "안녕하세요!",
        "고객센터에 연결해드리겠습니다.",
        "OTP 번호를 확인해주세요.",
        "공인인증서를 갱신해주시기 바랍니다."
    ]
    
    for i, test_text in enumerate(test_texts, 1):
        try:
            print(f"\n테스트 {i}: '{test_text}'")
            audio_data = await tts.generate_speech(test_text)
            
            # 파일로 저장
            filename = f"test_tts_{i}.mp3"
            with open(filename, "wb") as f:
                f.write(audio_data)
            
            print(f"✅ 성공! 파일 저장: {filename} ({len(audio_data)} bytes)")
            
        except Exception as e:
            print(f"❌ 실패: {str(e)}")
    
    await tts.cleanup()
    print("\n🎉 TTS 테스트 완료!")

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_tts_service())