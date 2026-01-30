제공해주신 정보를 바탕으로, 프로젝트의 기술적 가치와 가독성을 높인 README.md 양식으로 정리해 드립니다. 특히 데이터베이스 스키마와 인프라 설정 부분을 명확히 구분하여 협업자가 이해하기 쉽게 구성했습니다.📖 IP.SUM AI RangeGraph ServerKT 에이블스쿨 8기 빅프로젝트 AI 17조 > Gemini API 기반의 AI Agent를 활용한 웹소설 세계관 관리 및 AI 보조 서버입니다.Frontend Repository | Backend Repository🚀 프로젝트 개요IP.SUM은 웹소설 작가들을 위한 세계관 관리 도구입니다. Gemini API를 통해 소설 내용을 분석하고, 설정(Lorebook)을 자동으로 추출하거나 유사도 기반의 설정을 검색하는 AI Agent 기능을 제공합니다.🛠 기술 스택Language: Python 3.10+Framework: FastAPIAI SDK: Google Generative AI (Gemini)Database: PostgreSQL 14+ (with pgvector extension)📋 사전 준비 사항 (Prerequisites)1. Database 설정 (PostgreSQL + pgvector)본 프로젝트는 벡터 유사도 검색을 위해 pgvector 확장이 필요합니다. 아래 SQL 스크립트를 순서대로 실행하여 스키마를 생성하세요.<details><summary><b>📐 Database Schema SQL 보기 (클릭)</b></summary>SQL-- 1. 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 테이블 생성 (Universes, Works, Authors, Episodes, Lorebooks)
-- [전체 SQL 코드는 본문의 내용을 그대로 유지하여 실행해 주세요]

-- 3. 벡터 인덱스 최적화 (HNSW 사용)
CREATE INDEX idx_lorebooks_embedding ON lorebooks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
</details>2. 환경 설정 파일 생성서버 구동을 위해 다음 세 가지 파일을 프로젝트 루트 디렉토리에 생성해야 합니다.🔑 api_key.txt (Gemini API 키)인덱스 번호별로 API 키를 관리합니다. (부하 분산 및 백업용)Plaintext0=YOUR_GEMINI_API_KEY_1
1=YOUR_GEMINI_API_KEY_2
🗄️ db_info.txt (데이터베이스 연결 정보)Plaintexthost=your_db_host_ip
database=your_db_name
user=your_db_username
password=your_db_password
port=5432
🏗 데이터 구조 (Core Tables)테이블명설명비고Universes대규모 세계관 단위 정보 저장모든 작품의 상위 개념Works개별 소설 작품 정보장르, 시놉시스 등 포함Episodes작품별 회차 정보텍스트 경로 및 분석 여부 관리Lorebooks핵심 설정 정보벡터 임베딩을 통한 유사도 검색 가능🤖 AI 기능 활용Gemini AI Agent: 소설 본문을 분석하여 인물, 장소, 아이템 설정을 자동 추출합니다.Vector Search: 작성 중인 내용과 관련된 설정을 pgvector의 코사인 유사도 검색을 통해 실시간으로 추천합니다.💻 실행 방법필요한 라이브러리 설치: pip install -r requirements.txt환경 설정 파일 준비 (api_key.txt, db_info.txt)서버 실행: uvicorn main:app --reload
