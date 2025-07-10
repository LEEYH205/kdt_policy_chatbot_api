# 정책 챗봇 API

한국 지역 정책 검색 및 추천을 위한 AI 챗봇 API 패키지입니다.

## 설치 방법

```bash
pip install --no-cache-dir git+https://github.com/LEEYH205/kdt_policy_chatbot_api.git@v1.0.9
```

## 실행 방법

```bash
policy-api
```

## 웹 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 주요 엔드포인트

- `/health` : 서버 및 모델 상태 확인
- `/search` : 정책 검색 (POST, JSON body)
- `/search/simple` : 간단 검색 (GET)
- `/summary` : 정책 요약 (POST)
- `/regions` : 지원 지역 목록 조회 (GET)

## 예시: 정책 검색 요청 (POST)

```python
import requests

url = "http://localhost:8000/search"
payload = {
    "query": "중소기업 기술지원",
    "top_k": 5,
    "similarity_threshold": 0,
    "region_filter": "포천시",
    "target_filter": "중소기업",
    "field_filter": "기술개발",
    "region_weight": 0.3,
    "target_weight": 0.2,
    "field_weight": 0.2
}
response = requests.post(url, json=payload)
print(response.json())
```


### 검색 결과가 없는 경우
- 검색어 변경
- 필터 조건 완화
- 유사도 임계값 조정

## 성능 지표

- **응답 시간**: 평균 40ms
- **처리량**: 초당 10-50 요청
- **메모리 사용량**: 약 2-4GB
- **정확도**: 유사도 점수 0.7+ 기준

## 문의
- Author: KDT Hackathon Team (B2A5) @lyh
- Email: ejrdkachry@gmail.com 