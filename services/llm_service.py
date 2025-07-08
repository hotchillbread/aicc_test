import asyncio
import time
import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️ openai가 설치되지 않음")
    OPENAI_AVAILABLE = False

# 한국어 LLM (HuggingFace)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    print("⚠️ transformers/torch가 설치되지 않음")
    HUGGINGFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.openai_client = None
        self.korean_llm = None
        self.korean_tokenizer = None
        self.korean_model = None
        self.current_korean_model = None  # 현재 사용 중인 모델명
        self.device = "cpu"  # CPU 사용
        
    async def initialize(self):
        """LLM 모델들 초기화"""
        logger.info("LLM 서비스 초기화 중...")
        
        # OpenAI 초기화
        if OPENAI_AVAILABLE:
            try:
                # 환경변수에서 API 키 읽기
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.openai_client = openai.OpenAI(api_key=api_key)
                    logger.info("✅ OpenAI 클라이언트 초기화 완료")
                else:
                    logger.warning("⚠️ OPENAI_API_KEY 환경변수가 설정되지 않음")
            except Exception as e:
                logger.error(f"❌ OpenAI 초기화 실패: {str(e)}")
        
        # 한국어 LLM 초기화
        if HUGGINGFACE_AVAILABLE:
            try:
                logger.info("🇰🇷 한국어 LLM 모델 로드 중...")
                
                # 한국어 특화 모델들 우선순위별로 시도
                korean_models = [
                    "microsoft/DialoGPT-small"            # 한국어 BERT
                ]
                
                for model_name in korean_models:
                    try:
                        logger.info(f"모델 시도 중: {model_name}")
                        
                        self.korean_llm = pipeline(
                            "text-generation",
                            model=model_name,
                            device=-1,  # CPU 사용
                            framework="pt"
                        )
                        
                        # 간단한 테스트
                        test_result = self.korean_llm("안녕", max_length=50, num_return_sequences=1)
                        if test_result and len(test_result) > 0:
                            logger.info(f"✅ {model_name} 모델 로드 및 테스트 성공")
                            self.current_korean_model = model_name
                            break
                        
                    except Exception as model_error:
                        logger.warning(f"❌ {model_name} 실패: {str(model_error)}")
                        continue
                
                if not self.korean_llm:
                    logger.error("모든 한국어 모델 로드 실패")
                    
            except Exception as e:
                logger.error(f"❌ 한국어 LLM 초기화 실패: {str(e)}")
                self.korean_llm = None
        
        logger.info("🧠 LLM 서비스 초기화 완료!")
    
    async def generate_openai(self, prompt: str) -> Dict[str, Any]:
        """OpenAI GPT로 응답 생성"""
        start_time = time.time()
        
        if not self.openai_client:
            return {
                "model": "openai_gpt",
                "text": "",
                "error": "OpenAI 클라이언트가 초기화되지 않음",
                "processing_time": 0
            }
        
        try:
            # 한국어 고객센터 맥락 추가
            system_prompt = """당신은 한국의 금융 고객센터 상담원입니다. 
고객의 문의에 대해 친절하고 정확하게 답변해주세요.
답변은 간결하고 이해하기 쉽게 해주세요."""
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._openai_generate_sync,
                system_prompt, prompt
            )
            
            processing_time = time.time() - start_time
            
            return {
                "model": "openai_gpt",
                "text": response.choices[0].message.content.strip(),
                "tokens_used": response.usage.total_tokens,
                "processing_time": round(processing_time, 2),
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"OpenAI 생성 오류: {str(e)}")
            return {
                "model": "openai_gpt",
                "text": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _openai_generate_sync(self, system_prompt: str, user_prompt: str):
        """OpenAI 동기 처리"""
        return self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # 또는 "gpt-4"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
    
    async def generate_korean(self, prompt: str) -> Dict[str, Any]:
        """한국어 LLM으로 응답 생성"""
        start_time = time.time()
        
        if not self.korean_llm:
            return {
                "model": "korean_llm",
                "text": "",
                "error": "한국어 LLM이 초기화되지 않음",
                "processing_time": 0
            }
        
        try:
            # 간단한 프롬프트
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._korean_llm_generate_sync,
                prompt
            )
            
            processing_time = time.time() - start_time
            
            # 응답 텍스트 정리
            if response and len(response) > 0:
                generated_text = response[0]["generated_text"]
                # 원본 프롬프트 제거하고 답변 부분만 추출
                answer = generated_text.replace(prompt, "").strip()
                if not answer:  # 답변이 비어있으면 전체 텍스트 사용
                    answer = generated_text.strip()
            else:
                answer = "응답 생성 실패"
            
            return {
                "model": "korean_llm",
                "text": answer,
                "processing_time": round(processing_time, 2),
                "full_response": response[0]["generated_text"] if response else "None"
            }
            
        except Exception as e:
            logger.error(f"한국어 LLM 생성 오류: {str(e)}")
            return {
                "model": "korean_llm",
                "text": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _korean_llm_generate_sync(self, prompt: str):
        """한국어 LLM 동기 처리"""
        try:
            result = self.korean_llm(
                prompt,
                max_new_tokens=200,
                temperature=0.7,          
                do_sample=True,
                truncation=True
            )
            return result
        except Exception as e:
            logger.error(f"한국어 LLM 생성 오류: {e}")
            # 빈 응답 반환 (더미 없이)
            return [{"generated_text": prompt}]
    
    async def cleanup(self):
        """리소스 정리"""
        logger.info("LLM 서비스 리소스 정리 중...")
        
        if self.korean_model:
            del self.korean_model
            self.korean_model = None
            
        if self.korean_tokenizer:
            del self.korean_tokenizer
            self.korean_tokenizer = None
            
        if self.korean_llm:
            del self.korean_llm
            self.korean_llm = None
            
        # GPU 메모리 정리 (사용하는 경우)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("✅ LLM 리소스 정리 완료")

# 비교 테스트용 함수
async def compare_llm_models(prompt: str):
    """두 LLM 모델 비교"""
    llm = LLMService()
    await llm.initialize()
    
    print(f"🧠 프롬프트: {prompt}")
    print("=" * 50)
    
    # 병렬 처리
    tasks = [
        llm.generate_openai(prompt),
        llm.generate_korean(prompt)
    ]
    
    results = await asyncio.gather(*tasks)
    openai_result, korean_result = results
    
    print("🌐 OpenAI GPT:")
    print(f"   응답: {openai_result.get('text', 'ERROR')}")
    print(f"   토큰: {openai_result.get('tokens_used', 'N/A')}")
    print(f"   처리시간: {openai_result.get('processing_time', 0)}초")
    
    print("\n🇰🇷 한국어 LLM:")
    print(f"   응답: {korean_result.get('text', 'ERROR')}")
    print(f"   처리시간: {korean_result.get('processing_time', 0)}초")
    
    await llm.cleanup()
    return results

# 테스트용
async def test_llm_service():
    """LLM 서비스 테스트"""
    print("🧠 LLM 서비스 테스트 중...")
    
    llm = LLMService()
    await llm.initialize()
    
    print("✅ LLM 초기화 성공!")
    
    # 테스트 프롬프트들
    test_prompts = [
        "고객 문의에 답변해주세요: 공인인증서가 만료되었습니다. 갱신 방법을 알려주세요. ",
    ]
    
    for prompt in test_prompts:
        print(f"\n🔍 테스트: {prompt}")
        await compare_llm_models(prompt)
        print("-" * 30)
    
    await llm.cleanup()

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_llm_service())