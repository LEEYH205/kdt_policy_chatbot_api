# Policy Chatbot API

한국 지역 정책 검색을 위한 RESTful API 패키지입니다.

## 🚀 빠른 시작

### 설치

```bash
# Git에서 직접 설치
pip install git+https://github.com/LEEYH205/kdt_policy_chatbot_api.git
```

### API 서버 실행

```bash
# 기본 포트(8000)로 실행
policy-api

# 특정 포트로 실행
policy-api --port 8080

# 모든 IP에서 접근 허용
policy-api --host 0.0.0.0 --port 8000
```

### API 사용 예시

```python
import requests

# 정책 검색
response = requests.get("http://localhost:8000/search/simple", 
                       params={"query": "창업 지원", "top_k": 5})
results = response.json()

# 상세 검색
response = requests.post("http://localhost:8000/search", 
                        json={
                            "query": "포천시 창업 지원",
                            "top_k": 3,
                            "region_filter": "포천시"
                        })
results = response.json()
```

## 📋 API 엔드포인트

### 기본 검색
- `GET /search/simple` - 간단한 정책 검색
- `POST /search` - 상세한 정책 검색
- `POST /summary` - 정책 요약
- `GET /regions` - 사용 가능한 지역 목록
- `GET /health` - 서버 상태 확인

### API 문서
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔧 시스템 요구사항

- Python 3.8 이상
- pip
- 4GB 이상 RAM (모델 로딩용)

## 📞 지원

문제가 발생하면 이슈를 생성해주세요.

## 📄 라이선스

MIT License 