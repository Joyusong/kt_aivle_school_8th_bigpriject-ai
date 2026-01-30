# 📖 IP.SUM AI RangeGraph Server

> **KT 에이블스쿨 8기 빅프로젝트 AI 17조**
> Gemini API 기반의 AI Agent를 활용한 웹소설 세계관 관리 및 설정 추출 보조 서버입니다.

[Frontend Repository]((https://github.com/Joyusong/ai0917-kt-aivle-shool-8th-bigproject-frontend)) | [Backend Repository]((https://github.com/nsg716/ai0917-kt-aivle-school-8th-bigproject-backend))

---

## 🛠 Tech Stack

- **Framework:** FastAPI
- **AI Model:** Google Gemini API
- **Database:** PostgreSQL 14+ (with `pgvector` extension)
- **Library:** `psycopg2`, `google-generativeai`, `numpy` 등

---

## 📋 사전 준비 사항 (Prerequisites)

서버를 실행하기 전, 데이터베이스 스키마 구축과 환경 설정 파일 작성이 필요합니다.

### 1. Database 설정 (PostGres + VectorDB)

본 프로젝트는 설정값(Lorebook)의 유사도 검색을 위해 `pgvector` 확장을 사용합니다.

<details>
<summary>📂 <b>SQL 스키마 코드 보기/복사</b></summary>

```sql
-- vector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Universes (세계관)
CREATE TABLE universes ( 
    id BIGSERIAL PRIMARY KEY,
    owner_id BIGINT NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ
);

-- 2. Works (작품)
CREATE TABLE works (
    id BIGSERIAL PRIMARY KEY,
    universe_id BIGINT REFERENCES universes(id) ON DELETE SET NULL,
    primary_author_id BIGINT NOT NULL,
    title VARCHAR(255) NOT NULL,
    synopsis TEXT,
    genre VARCHAR(100),
    status VARCHAR(50) DEFAULT '연재중',
    cover_image_url VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ
);

-- 3. Work Authors (작가 및 공동저자)
CREATE TABLE work_authors (
    work_id BIGINT REFERENCES works(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    role VARCHAR(100) DEFAULT 'CO_AUTHOR',
    contribution_percent INT DEFAULT 0,
    joined_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (work_id, user_id)
);

-- 4. Episodes (회차)
CREATE TABLE episodes (
    id BIGSERIAL PRIMARY KEY,
    work_id BIGINT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    ep_num INT NOT NULL,
    title VARCHAR(255) NOT NULL,
    subtitle VARCHAR(255),
    txt_path VARCHAR(500) NOT NULL,
    word_count INT DEFAULT 0,
    is_read_only BOOLEAN DEFAULT FALSE,
    is_analyzed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ,
    UNIQUE (work_id, ep_num)
);

-- 5. Lorebooks (설정집/벡터 DB 포함)
CREATE TABLE lorebooks (
    id BIGSERIAL PRIMARY KEY,
    universe_id BIGINT REFERENCES universes(id) ON DELETE CASCADE,
    work_id BIGINT REFERENCES works(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    keyword VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,
    ep_num INT[],
    setting JSONB NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ
);

-- 인덱스 설정 (HNSW 벡터 검색 최적화)
CREATE INDEX idx_lorebooks_embedding ON lorebooks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
-- 기타 인덱스 및 트리거 생략 (전체 쿼리는 스키마 파일 참조)
