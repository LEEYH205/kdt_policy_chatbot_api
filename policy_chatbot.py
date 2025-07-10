import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Dict, Tuple
import re
from numpy.linalg import norm

class PolicyChatbot:
    def __init__(self, csv_path: str = "./data/gyeonggi_smallbiz_policies_2000_소상공인,경기_20250705.csv", model_name: str = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"):
        """
        정책 챗봇 초기화
        
        Args:
            csv_path: 정책 데이터 CSV 파일 경로
            model_name: 임베딩 모델명
        """
        self.csv_path = csv_path
        self.model_name = model_name
        self.data = None
        self.embeddings = None
        self.index = None
        self.model = None
        
        # 데이터 로드 및 모델 초기화
        self._load_data()
        self._initialize_model()
        self._create_embeddings()
        
        # 지역 계층 구조 정의
        self.region_hierarchy = {
            # 전국, 서울, 경기만
            # 경기도 하위 지역들
            "포천시": ["포천시", "경기도", "전국"],
            "가평군": ["가평군", "경기도", "전국"],
            "양평군": ["양평군", "경기도", "전국"],
            "여주시": ["여주시", "경기도", "전국"],
            "이천시": ["이천시", "경기도", "전국"],
            "용인시": ["용인시", "경기도", "전국"],
            "안성시": ["안성시", "경기도", "전국"],
            "평택시": ["평택시", "경기도", "전국"],
            "오산시": ["오산시", "경기도", "전국"],
            "안산시": ["안산시", "경기도", "전국"],
            "시흥시": ["시흥시", "경기도", "전국"],
            "군포시": ["군포시", "경기도", "전국"],
            "의왕시": ["의왕시", "경기도", "전국"],
            "안양시": ["안양시", "경기도", "전국"],
            "과천시": ["과천시", "경기도", "전국"],
            "광명시": ["광명시", "경기도", "전국"],
            "부천시": ["부천시", "경기도", "전국"],
            "김포시": ["김포시", "경기도", "전국"],
            "고양시": ["고양시", "경기도", "전국"],
            "파주시": ["파주시", "경기도", "전국"],
            "연천군": ["연천군", "경기도", "전국"],
            "동두천시": ["동두천시", "경기도", "전국"],
            "의정부시": ["의정부시", "경기도", "전국"],
            "남양주시": ["남양주시", "경기도", "전국"],
            "구리시": ["구리시", "경기도", "전국"],
            "하남시": ["하남시", "경기도", "전국"],
            "성남시": ["성남시", "경기도", "전국"],
            "수원시": ["수원시", "경기도", "전국"],
            # 서울특별시 하위 지역들
            "강남구": ["강남구", "서울특별시", "전국"],
            "강동구": ["강동구", "서울특별시", "전국"],
            "강북구": ["강북구", "서울특별시", "전국"],
            "강서구": ["강서구", "서울특별시", "전국"],
            "관악구": ["관악구", "서울특별시", "전국"],
            "광진구": ["광진구", "서울특별시", "전국"],
            "구로구": ["구로구", "서울특별시", "전국"],
            "금천구": ["금천구", "서울특별시", "전국"],
            "노원구": ["노원구", "서울특별시", "전국"],
            "도봉구": ["도봉구", "서울특별시", "전국"],
            "동대문구": ["동대문구", "서울특별시", "전국"],
            "동작구": ["동작구", "서울특별시", "전국"],
            "마포구": ["마포구", "서울특별시", "전국"],
            "서대문구": ["서대문구", "서울특별시", "전국"],
            "서초구": ["서초구", "서울특별시", "전국"],
            "성동구": ["성동구", "서울특별시", "전국"],
            "성북구": ["성북구", "서울특별시", "전국"],
            "송파구": ["송파구", "서울특별시", "전국"],
            "양천구": ["양천구", "서울특별시", "전국"],
            "영등포구": ["영등포구", "서울특별시", "전국"],
            "용산구": ["용산구", "서울특별시", "전국"],
            "은평구": ["은평구", "서울특별시", "전국"],
            "종로구": ["종로구", "서울특별시", "전국"],
            "중구": ["중구", "서울특별시", "전국"],
            "중랑구": ["중랑구", "서울특별시", "전국"],
            # 상위 지역들
            "경기도": ["경기도", "전국"],
            "서울특별시": ["서울특별시", "전국"],
            "부산광역시": ["부산광역시", "전국"],
            "인천광역시": ["인천광역시", "전국"],
            "대구광역시": ["대구광역시", "전국"],
            "광주광역시": ["광주광역시", "전국"],
            "대전광역시": ["대전광역시", "전국"],
            "울산광역시": ["울산광역시", "전국"],
            "세종특별자치시": ["세종특별자치시", "전국"],
            "강원도": ["강원도", "전국"],
            "충청북도": ["충청북도", "전국"],
            "충청남도": ["충청남도", "전국"],
            "전라북도": ["전라북도", "전국"],
            "전라남도": ["전라남도", "전국"],
            "경상북도": ["경상북도", "전국"],
            "경상남도": ["경상남도", "전국"],
            "제주특별자치도": ["제주특별자치도", "전국"],
            # 최상위 지역
            "전국": ["전국"]
        }
        
    def _load_data(self):
        """CSV 데이터 로드 및 전처리"""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"데이터 로드 완료: {len(self.data)}개 정책")
            
            # 결측값 처리
            self.data = self.data.fillna("")
            
            # 텍스트 전처리
            self.data['processed_text'] = self.data.apply(self._preprocess_text, axis=1)
            
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            raise
    
    def _preprocess_text(self, row: pd.Series) -> str:
        """텍스트 전처리"""
        # 주요 필드들을 결합하여 검색용 텍스트 생성
        fields = [
            str(row['title(공고명)']),
            str(row['body_text(공고내용)']),
            str(row['지원대상']),
            str(row['소관기관']),
            str(row['지원분야(대)']),
            str(row['지원분야(중)']),
            str(row['사업수행기관']),
            str(row['문의처']),
            str(row['신청기간']),
            str(row['사업신청방법설명'])
        ]
        
        # 텍스트 결합 및 정리
        combined_text = " ".join(fields)
        
        # 특수문자 제거 및 공백 정리
        combined_text = re.sub(r'[^\w\s가-힣]', ' ', combined_text)
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()
        
        return combined_text
    
    def _initialize_model(self):
        """임베딩 모델 초기화"""
        try:
            print("임베딩 모델 로딩 중...")
            self.model = SentenceTransformer(self.model_name)
            print("모델 로딩 완료")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            # 한국어에 특화된 모델로 대체
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    def _create_embeddings(self):
        """텍스트 임베딩 생성 및 FAISS 인덱스 구축"""
        try:
            print("임베딩 생성 중...")
            
            # 텍스트 임베딩 생성
            texts = self.data['processed_text'].tolist()
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # FAISS 인덱스 구축
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
            
            # 정규화 (cosine similarity를 위해)
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype('float32'))
            
            print(f"임베딩 생성 완료: {len(self.embeddings)}개 벡터")
            
        except Exception as e:
            print(f"임베딩 생성 실패: {e}")
            raise
    
    def search_policies(self, query, top_k=5, similarity_threshold=0.0, region_filter=None, target_filter=None, field_filter=None, region_weight=0.3, target_weight=0.2, field_weight=0.2):
        query_emb = self.model.encode(query)
        query_emb = np.array(query_emb).reshape(1, -1)
        # FAISS에서 모든 벡터 가져오기
        all_embs = self.index.reconstruct_n(0, self.data.shape[0])
        # 코사인 유사도 계산 (0~1)
        def cosine_similarity(a, b):
            return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)
        sim_scores = np.array([cosine_similarity(query_emb[0], emb) for emb in all_embs])
        # 내림차순 정렬 인덱스
        sorted_idx = np.argsort(sim_scores)[::-1]
        results = []
        for idx in sorted_idx:
            row = self.data.iloc[idx]
            # 하드 필터 적용 (지역: 정책명/본문에 지역명 명시 여부까지 반영)
            if region_filter:
                org = str(row.get('소관기관', ''))
                title = str(row.get('title(공고명)', ''))
                body = str(row.get('body_text(공고내용)', ''))
                # 1. 소관기관이 region_filter(포천시)면 무조건 포함
                if org == region_filter:
                    pass
                # 2. 소관기관이 region_filter의 상위(경기도) 또는 전국이면, title/body에 region_filter가 명시되어야 포함
                elif region_filter in self.region_hierarchy and org in self.region_hierarchy[region_filter][1:]:
                    if region_filter not in title and region_filter not in body:
                        continue
                # 3. 그 외(다른 시/군)는 제외
                else:
                    continue
            if target_filter and target_filter not in str(row.get('지원대상', '')):
                continue
            if field_filter and field_filter not in str(row.get('지원분야(대)', '')):
                continue
            filter_score = 0.0
            # 지역명 가중치 제거 (region_weight 관련 코드 삭제)
            if target_filter:
                filter_score += target_weight
            if field_filter:
                filter_score += field_weight
            final_score = sim_scores[idx] + filter_score
            if final_score >= similarity_threshold:
                results.append({
                    'title': row.get('title(공고명)', ''),
                    'body': row.get('body_text(공고내용)', ''),
                    'target': row.get('지원대상', ''),
                    'organization': row.get('소관기관', ''),
                    'field_major': row.get('지원분야(대)', ''),
                    'field_minor': row.get('지원분야(중)', ''),
                    'executing_org': row.get('사업수행기관', ''),
                    'contact': row.get('문의처', ''),
                    'period': row.get('신청기간', ''),
                    'application_method': row.get('사업신청방법설명', ''),
                    'similarity_score': final_score
                })
            if len(results) >= top_k:
                break
        return results
    
    def get_policy_summary(self, query: str) -> str:
        """정책 요약 정보 생성"""
        results = self.search_policies(query, top_k=3)
        
        if not results:
            return "관련 정책을 찾을 수 없습니다."
        
        summary = f"'{query}'와 관련된 정책을 찾았습니다:\n\n"
        
        for result in results:
            summary += f"📋 {result['title']}\n"
            summary += f"🎯 지원대상: {result['target']}\n"
            summary += f"🏢 소관기관: {result['organization']}\n"
            summary += f"📅 신청기간: {result['period']}\n"
            summary += f"📞 문의처: {result['contact']}\n"
            summary += f"📝 신청방법: {result['application_method'][:100]}...\n"
            summary += f"📊 유사도 점수: {result['similarity_score']:.3f}\n"
            summary += "-" * 50 + "\n"
        
        return summary
    
    def save_model(self, path: str = "policy_chatbot_model.pkl"):
        """모델 저장"""
        try:
            model_data = {
                'data': self.data,
                'embeddings': self.embeddings,
                'index': self.index,
                'model_name': self.model_name
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"모델 저장 완료: {path}")
            
        except Exception as e:
            print(f"모델 저장 실패: {e}")
    
    def load_model(self, path: str = "policy_chatbot_model.pkl"):
        """모델 로드"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.data = model_data['data']
            self.embeddings = model_data['embeddings']
            self.index = model_data['index']
            self.model_name = model_data['model_name']
            
            # 모델 재초기화
            self._initialize_model()
            
            print(f"모델 로드 완료: {path}")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")

# 사용 예시
if __name__ == "__main__":
    # 챗봇 초기화
    chatbot = PolicyChatbot()
    
    # 테스트 검색
    test_queries = [
        "중소기업 기술지원",
        "창업 지원",
        "수출 진출",
        "청년 지원",
        "AI 기술 개발"
    ]
    
    for query in test_queries:
        print(f"\n🔍 검색어: {query}")
        print("=" * 50)
        results = chatbot.search_policies(query, top_k=3)
        
        for result in results:
            print(f"📋 {result['title']}")
            print(f"🎯 {result['target']} | 📊 유사도: {result['similarity_score']:.3f}")
            print("-" * 30) 