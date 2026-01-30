# 📖 IP.SUM AI RangeGraph Server

> **KT 에이블스쿨 8기 빅프로젝트 AI 17조**
> Gemini API 기반의 AI Agent를 활용한 웹소설 세계관 관리 및 설정 추출 보조 서버입니다.

[Frontend Repository](https://github.com/Joyusong/ai0917-kt-aivle-shool-8th-bigproject-frontend) | [Backend Repository](https://github.com/nsg716/ai0917-kt-aivle-school-8th-bigproject-backend)

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
-- ========================================
-- IP.SUM Integrated Database Schema
-- PostgreSQL 14+ with Standard Naming (snake_case)
-- ========================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ========================================
-- Core: Users & Authentication
-- ========================================

CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',
    integration_id VARCHAR(255) UNIQUE,
    last_activity_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_integration_id ON users(integration_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);

COMMENT ON TABLE users IS '사용자 테이블';
COMMENT ON COLUMN users.integration_id IS '외부 인증 시스템 ID (Auth0, Supabase 등)';

-- ========================================
-- Core: Invite Codes
-- ========================================

CREATE TABLE invite_codes (
    id BIGSERIAL PRIMARY KEY,
    author_integration_id VARCHAR(255) REFERENCES users(integration_id) ON DELETE CASCADE,
    code VARCHAR(100) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_invite_codes_author ON invite_codes(author_integration_id);
CREATE INDEX idx_invite_codes_expires ON invite_codes(expires_at);

COMMENT ON TABLE invite_codes IS '초대 코드 테이블';

-- ========================================
-- Novel Platform: Universes
-- ========================================

CREATE TABLE universes (
    id BIGSERIAL PRIMARY KEY,
    owner_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ
);

CREATE INDEX idx_universes_owner ON universes(owner_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_universes_deleted ON universes(id) WHERE deleted_at IS NULL;

COMMENT ON TABLE universes IS '유니버스 - 여러 작품을 묶는 세계관';

-- ========================================
-- Novel Platform: Works
-- ========================================

CREATE TABLE works (
    id BIGSERIAL PRIMARY KEY,
    universe_id BIGINT REFERENCES universes(id) ON DELETE SET NULL,
    primary_author_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    synopsis TEXT,
    genre VARCHAR(100),
    status VARCHAR(50) DEFAULT '연재중',
    cover_image_url VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ
);

CREATE INDEX idx_works_author ON works(primary_author_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_works_universe ON works(universe_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_works_status ON works(status) WHERE deleted_at IS NULL;

COMMENT ON TABLE works IS '작품 테이블';

-- ========================================
-- Novel Platform: Work Authors
-- ========================================

CREATE TABLE work_authors (
    work_id BIGINT REFERENCES works(id) ON DELETE CASCADE,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(100) DEFAULT 'CO_AUTHOR',
    contribution_percent INT DEFAULT 0,
    joined_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (work_id, user_id)
);

CREATE INDEX idx_work_authors_user ON work_authors(user_id);
CREATE INDEX idx_work_authors_work ON work_authors(work_id);

COMMENT ON TABLE work_authors IS '공동 저자 관계 테이블';

-- ========================================
-- Novel Platform: Episodes
-- ========================================

CREATE TABLE episodes (
    id BIGSERIAL PRIMARY KEY,
    work_id BIGINT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
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

CREATE INDEX idx_episodes_work ON episodes(work_id, ep_num) WHERE deleted_at IS NULL;
CREATE INDEX idx_episodes_user ON episodes(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_episodes_analyzed ON episodes(work_id, is_analyzed) WHERE deleted_at IS NULL;

COMMENT ON TABLE episodes IS '에피소드(회차) 테이블';

-- ========================================
-- Novel Platform: Lorebooks
-- ========================================

CREATE TABLE lorebooks (
    id BIGSERIAL PRIMARY KEY,
    universe_id BIGINT REFERENCES universes(id) ON DELETE CASCADE,
    work_id BIGINT REFERENCES works(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    keyword VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,
    ep_num INT[],
    setting JSONB NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ
);

CREATE INDEX idx_lorebooks_universe ON lorebooks(universe_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_lorebooks_work ON lorebooks(work_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_lorebooks_user ON lorebooks(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_lorebooks_keyword ON lorebooks(keyword) WHERE deleted_at IS NULL;
CREATE INDEX idx_lorebooks_category ON lorebooks(category) WHERE deleted_at IS NULL;
CREATE INDEX idx_lorebooks_setting ON lorebooks USING GIN (setting);
CREATE INDEX idx_lorebooks_ep_num ON lorebooks USING GIN (ep_num);
CREATE INDEX idx_lorebooks_embedding ON lorebooks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

COMMENT ON TABLE lorebooks IS '설정집 테이블';

-- ========================================
-- Admin: Admin Notices
-- ========================================

CREATE TABLE admin_notices (
    id BIGSERIAL PRIMARY KEY,
    action_url VARCHAR(500),
    category VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_read BOOLEAN DEFAULT FALSE,
    message TEXT,
    metadata TEXT,
    related_entity VARCHAR(255),
    severity VARCHAR(50),
    source VARCHAR(100),
    target_role VARCHAR(50)
);

CREATE INDEX idx_admin_notices_category ON admin_notices(category);
CREATE INDEX idx_admin_notices_created ON admin_notices(created_at);
CREATE INDEX idx_admin_notices_is_read ON admin_notices(is_read);
CREATE INDEX idx_admin_notices_source ON admin_notices(source);
CREATE INDEX idx_admin_notices_target ON admin_notices(target_role);

COMMENT ON TABLE admin_notices IS '관리자 알림 테이블';

-- ========================================
-- Admin: System Metrics
-- ========================================

CREATE TABLE system_metrics (
    id BIGSERIAL PRIMARY KEY,
    cpu_usage DOUBLE PRECISION,
    memory_usage DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp);

COMMENT ON TABLE system_metrics IS '시스템 메트릭 테이블';

-- ========================================
-- Admin: System Logs
-- ========================================

CREATE TABLE system_logs (
    id BIGSERIAL PRIMARY KEY,
    level VARCHAR(50),
    message TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_read BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX idx_system_logs_level ON system_logs(level);

COMMENT ON TABLE system_logs IS '시스템 로그 테이블';

-- ========================================
-- Admin: IP Trend Reports
-- ========================================

CREATE TABLE ip_trend_reports (
    id BIGSERIAL PRIMARY KEY,
    analysis_date TIMESTAMPTZ NOT NULL,
    file_path VARCHAR(500),
    status VARCHAR(50)
);

CREATE INDEX idx_ip_trend_reports_date ON ip_trend_reports(analysis_date);

COMMENT ON TABLE ip_trend_reports IS 'IP 트렌드 리포트 테이블';

-- ========================================
-- Admin: Deployment Info
-- ========================================

CREATE TABLE deployment_info (
    id BIGSERIAL PRIMARY KEY,
    deployment_time TIMESTAMPTZ NOT NULL,
    version VARCHAR(100),
    status VARCHAR(50)
);

CREATE INDEX idx_deployment_info_time ON deployment_info(deployment_time);

COMMENT ON TABLE deployment_info IS '배포 정보 테이블';

-- ========================================
-- Admin: Notices
-- ========================================

CREATE TABLE notices (
    id BIGSERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_notices_created ON notices(created_at);

COMMENT ON TABLE notices IS '공지사항 테이블';

-- ========================================
-- Admin: Daily Active Users
-- ========================================

CREATE TABLE daily_active_users (
    id BIGSERIAL PRIMARY KEY,
    count INT NOT NULL DEFAULT 0,
    date TIMESTAMPTZ UNIQUE NOT NULL
);

CREATE INDEX idx_daily_active_users_date ON daily_active_users(date);

COMMENT ON TABLE daily_active_users IS '일별 활성 사용자 통계';

-- ========================================
-- Triggers: updated_at auto-update
-- ========================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER universes_updated_at BEFORE UPDATE ON universes FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER works_updated_at BEFORE UPDATE ON works FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER episodes_updated_at BEFORE UPDATE ON episodes FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER lorebooks_updated_at BEFORE UPDATE ON lorebooks FOR EACH ROW EXECUTE FUNCTION update_updated_at();

COMMENT ON FUNCTION update_updated_at() IS 'updated_at 자동 갱신 트리거 함수';

-- ========================================
-- Triggers: Episode deletion sync
-- ========================================

CREATE OR REPLACE FUNCTION sync_lorebook_on_episode_delete()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE lorebooks
    SET ep_num = array_remove(ep_num, OLD.ep_num)
    WHERE work_id = OLD.work_id AND OLD.ep_num = ANY(ep_num) AND deleted_at IS NULL;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER episode_delete_sync
    BEFORE UPDATE OF deleted_at ON episodes
    FOR EACH ROW
    WHEN (OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL)
    EXECUTE FUNCTION sync_lorebook_on_episode_delete();

COMMENT ON FUNCTION sync_lorebook_on_episode_delete() IS 'Episode 삭제 시 Lorebook 배열 동기화';

-- ========================================
-- Views: Active records (soft delete)
-- ========================================

CREATE VIEW active_universes AS SELECT * FROM universes WHERE deleted_at IS NULL;
CREATE VIEW active_works AS SELECT * FROM works WHERE deleted_at IS NULL;
CREATE VIEW active_episodes AS SELECT * FROM episodes WHERE deleted_at IS NULL;
CREATE VIEW active_lorebooks AS SELECT * FROM lorebooks WHERE deleted_at IS NULL;

COMMENT ON VIEW active_universes IS '활성 유니버스 뷰';
COMMENT ON VIEW active_works IS '활성 작품 뷰';
COMMENT ON VIEW active_episodes IS '활성 에피소드 뷰';
COMMENT ON VIEW active_lorebooks IS '활성 설정집 뷰';

-- ========================================
-- Sample Data
-- ========================================

-- Users
INSERT INTO users (email, name, role, integration_id) VALUES
('author1@ipsum.com', '작가1', 'author', 'auth0|1001'),
('author2@ipsum.com', '작가2', 'author', 'auth0|1002'),
('admin@ipsum.com', '관리자', 'admin', 'auth0|9999');

-- Universes
INSERT INTO universes (owner_id, title, description) VALUES
(1, '룬테라 세계관', '마법과 룬이 지배하는 판타지 세계');

-- Works
INSERT INTO works (universe_id, primary_author_id, title, synopsis, genre, status) VALUES
(1, 1, '그웬의 여정', '살아있는 인형 그웬의 이야기', '다크 판타지', '연재중'),
(1, 1, '비에고의 복수', '몰락한 왕 비에고의 이야기', '다크 판타지', '연재중');

-- Work Authors
INSERT INTO work_authors (work_id, user_id, role, contribution_percent) VALUES
(1, 2, 'CO_AUTHOR', 20);

-- Episodes
INSERT INTO episodes (work_id, user_id, ep_num, title, subtitle, txt_path, word_count, is_analyzed) VALUES
(1, 1, 1, '첫 번째 이야기', '각성', '/storage/works/1/ep1.txt', 3500, true),
(1, 1, 2, '두 번째 이야기', '기억', '/storage/works/1/ep2.txt', 4200, true),
(1, 1, 3, '세 번째 이야기', '결의', '/storage/works/1/ep3.txt', 3800, false);

-- Lorebooks
INSERT INTO lorebooks (universe_id, user_id, keyword, category, ep_num, setting) VALUES
(1, 1, '그웬', '인물', ARRAY[1, 2], '{"name":"그웬","type":"인형","abilities":["신성한 안개","마법 가위"]}'::jsonb),
(1, 1, '비에고', '인물', ARRAY[1], '{"name":"비에고","type":"언데드","abilities":["검은 안개","망령 소환"]}'::jsonb),
(1, 1, '축복의 빛 군도', '장소', ARRAY[1, 2], '{"name":"축복의 빛 군도","type":"저주받은 땅","regions":["헬리아","녹시"]}'::jsonb);

-- Invite Codes
INSERT INTO invite_codes (author_integration_id, code, expires_at) VALUES
('auth0|1001', 'WELCOME2024', CURRENT_TIMESTAMP + INTERVAL '30 days');

-- Daily Active Users
INSERT INTO daily_active_users (count, date) VALUES
(150, CURRENT_DATE),
(142, CURRENT_DATE - INTERVAL '1 day'),
(138, CURRENT_DATE - INTERVAL '2 days');

-- Admin Notices
INSERT INTO admin_notices (category, message, severity, source, target_role) VALUES
('system', '시스템 점검 예정', 'info', 'system', 'all'),
('security', '비정상 접근 감지', 'warning', 'security_monitor', 'admin');

-- System Metrics
INSERT INTO system_metrics (cpu_usage, memory_usage) VALUES
(45.2, 62.8),
(48.5, 65.1);

-- Notices
INSERT INTO notices (title, content) VALUES
('서비스 오픈 안내', 'IP.SUM 베타 서비스가 오픈되었습니다.'),
('업데이트 공지', '새로운 기능이 추가되었습니다.');
```
</details>

### 2. Configuration 파일 작성
루트 디렉토리에 다음 파일들을 생성해 주세요.

✅ api_key.txt
```Plaintext
0=YOUR_GEMINI_API_KEY_0
1=YOUR_GEMINI_API_KEY_1
```

✅ db_info.txt
```Plaintext
host=your_host_ip
database=your_db_name
user=your_username
password=your_password
port=5432
```
📋 requirements.txt
```
pip install -r requirements.txt
```
