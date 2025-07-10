from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from .policy_chatbot import PolicyChatbot
from policy_chatbot_api import __version__
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="정책 챗봇 API",
    description="정책 검색 및 추천을 위한 AI 챗봇 API",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 챗봇 인스턴스
chatbot = None

# Pydantic 모델들
class SearchRequest(BaseModel):
    query: str = Field(..., description="검색 쿼리", example="중소기업 기술지원")
    top_k: int = Field(default=5, ge=1, le=20, description="반환할 결과 수")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="유사도 임계값")
    region_filter: Optional[str] = Field(default=None, description="지역 필터", example="포천시")
    target_filter: Optional[str] = Field(default=None, description="지원대상 필터", example="중소기업")
    field_filter: Optional[str] = Field(default=None, description="지원분야 필터", example="기술개발")
    region_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="지역 가중치")
    target_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="지원대상 가중치")
    field_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="지원분야 가중치")

class PolicyResult(BaseModel):
    title: str = Field(..., description="정책 제목")
    body: str = Field(..., description="정책 내용")
    target: str = Field(..., description="지원대상")
    organization: str = Field(..., description="소관기관")
    field_major: str = Field(..., description="지원분야(대)")
    field_minor: str = Field(..., description="지원분야(중)")
    executing_org: str = Field(..., description="사업수행기관")
    contact: str = Field(..., description="문의처")
    period: str = Field(..., description="신청기간")
    application_method: str = Field(..., description="사업신청방법설명")
    similarity_score: float = Field(..., description="유사도 점수")

class SearchResponse(BaseModel):
    query: str = Field(..., description="검색 쿼리")
    total_results: int = Field(..., description="총 결과 수")
    results: List[PolicyResult] = Field(..., description="검색 결과")
    filters_applied: Dict[str, Any] = Field(..., description="적용된 필터")

class SummaryRequest(BaseModel):
    query: str = Field(..., description="요약할 쿼리", example="중소기업 기술지원")

class SummaryResponse(BaseModel):
    query: str = Field(..., description="요약 쿼리")
    summary: str = Field(..., description="정책 요약")

class HealthResponse(BaseModel):
    status: str = Field(..., description="서버 상태")
    model_loaded: bool = Field(..., description="모델 로드 상태")
    data_count: int = Field(..., description="데이터 개수")

# 앱 시작 시 챗봇 초기화
@app.on_event("startup")
async def startup_event():
    global chatbot
    try:
        logger.info("정책 챗봇 초기화 중...")
        chatbot = PolicyChatbot()
        logger.info("정책 챗봇 초기화 완료")
    except Exception as e:
        logger.error(f"정책 챗봇 초기화 실패: {e}")
        raise

# 헬스 체크 엔드포인트
@app.get("/health", response_model=HealthResponse, tags=["시스템"])
async def health_check():
    """서버 상태 및 모델 로드 상태 확인"""
    global chatbot
    
    if chatbot is None:
        return HealthResponse(
            status="error",
            model_loaded=False,
            data_count=0
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        data_count=len(chatbot.data) if chatbot.data is not None else 0
    )

# 정책 검색 엔드포인트
@app.post("/search", response_model=SearchResponse, tags=["검색"])
async def search_policies(request: SearchRequest):
    """정책 검색 API"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="챗봇이 초기화되지 않았습니다.")
    
    try:
        logger.info(f"검색 요청: {request.query}")
        
        results = chatbot.search_policies(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            region_filter=request.region_filter,
            target_filter=request.target_filter,
            field_filter=request.field_filter,
            region_weight=request.region_weight,
            target_weight=request.target_weight,
            field_weight=request.field_weight
        )
        
        # 필터 정보 구성
        filters_applied = {
            "region_filter": request.region_filter,
            "target_filter": request.target_filter,
            "field_filter": request.field_filter,
            "similarity_threshold": request.similarity_threshold,
            "weights": {
                "region_weight": request.region_weight,
                "target_weight": request.target_weight,
                "field_weight": request.field_weight
            }
        }
        
        return SearchResponse(
            query=request.query,
            total_results=len(results),
            results=results,
            filters_applied=filters_applied
        )
        
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"검색 중 오류가 발생했습니다: {str(e)}")

# 정책 요약 엔드포인트
@app.post("/summary", response_model=SummaryResponse, tags=["요약"])
async def get_policy_summary(request: SummaryRequest):
    """정책 요약 API"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="챗봇이 초기화되지 않았습니다.")
    
    try:
        logger.info(f"요약 요청: {request.query}")
        
        summary = chatbot.get_policy_summary(request.query)
        
        return SummaryResponse(
            query=request.query,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"요약 중 오류가 발생했습니다: {str(e)}")

# 간단한 검색 엔드포인트 (GET 요청)
@app.get("/search/simple", response_model=SearchResponse, tags=["검색"])
async def simple_search(
    query: str = Query(..., description="검색 쿼리"),
    top_k: int = Query(default=5, ge=1, le=20, description="반환할 결과 수"),
    region: Optional[str] = Query(default=None, description="지역 필터")
):
    """간단한 정책 검색 API (GET 요청)"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="챗봇이 초기화되지 않았습니다.")
    
    try:
        logger.info(f"간단 검색 요청: {query}")
        
        results = chatbot.search_policies(
            query=query,
            top_k=top_k,
            region_filter=region
        )
        
        filters_applied = {
            "region_filter": region,
            "target_filter": None,
            "field_filter": None,
            "similarity_threshold": 0.0,
            "weights": {
                "region_weight": 0.3,
                "target_weight": 0.2,
                "field_weight": 0.2
            }
        }
        
        return SearchResponse(
            query=query,
            total_results=len(results),
            results=results,
            filters_applied=filters_applied
        )
        
    except Exception as e:
        logger.error(f"간단 검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"검색 중 오류가 발생했습니다: {str(e)}")

# 사용 가능한 지역 목록 엔드포인트
@app.get("/regions", tags=["메타데이터"])
async def get_available_regions():
    """사용 가능한 지역 목록 반환"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="챗봇이 초기화되지 않았습니다.")
    
    try:
        regions = list(chatbot.region_hierarchy.keys())
        return {
            "regions": regions,
            "total_count": len(regions),
            "hierarchy": chatbot.region_hierarchy
        }
    except Exception as e:
        logger.error(f"지역 목록 조회 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"지역 목록 조회 중 오류가 발생했습니다: {str(e)}")

# 루트 엔드포인트
@app.get("/", tags=["시스템"])
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "정책 챗봇 API에 오신 것을 환영합니다!",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }

def create_app():
    """FastAPI 앱 인스턴스 반환"""
    return app

def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(description="정책 챗봇 API 서버")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--reload", action="store_true", help="자동 리로드 활성화")
    
    args = parser.parse_args()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 