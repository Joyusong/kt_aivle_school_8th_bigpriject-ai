## KT AIVLESCHOOL BIG PROJECT AI SERVER API

KT 에이블 스쿨 빅프로젝트 8기 AI 17조 빅프로젝트 AI RangeGraph 서버 파일입니다.

Gemini API를 사용하여 AI Agent를 구축하였습니다.

### 사전 준비

1. PostGres 의 VectorDB 생성
-- ========================================
-- IP.SUM Database Schema
-- PostgreSQL 14+ Compatible
-- ========================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ========================================
-- 1. Universes
-- ========================================
CREATE TABLE universes ( 
    id BIGSERIAL PRIMARY KEY,
    owner_id BIGINT NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ
);

-- ========================================
-- 2. Works
-- ========================================
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

-- ========================================
-- 3. Work Authors
-- ========================================
CREATE TABLE work_authors (
    work_id BIGINT REFERENCES works(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    role VARCHAR(100) DEFAULT 'CO_AUTHOR',
    contribution_percent INT DEFAULT 0,
    joined_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (work_id, user_id)
);

-- ========================================
-- 4. Episodes
-- ========================================
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

-- ========================================
-- 5. Lorebooks
-- ========================================
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

-- ========================================
-- Indexes
-- ========================================
CREATE INDEX idx_universes_owner ON universes(owner_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_works_author ON works(primary_author_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_works_universe ON works(universe_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_episodes_work ON episodes(work_id, ep_num) WHERE deleted_at IS NULL;
CREATE INDEX idx_episodes_user ON episodes(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_lorebooks_universe ON lorebooks(universe_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_lorebooks_work ON lorebooks(work_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_lorebooks_keyword ON lorebooks(keyword) WHERE deleted_at IS NULL;
CREATE INDEX idx_lorebooks_setting ON lorebooks USING GIN (setting);
CREATE INDEX idx_lorebooks_ep_num ON lorebooks USING GIN (ep_num);
CREATE INDEX idx_lorebooks_embedding ON lorebooks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- ========================================
-- Triggers
-- ========================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER universes_updated_at BEFORE UPDATE ON universes FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER works_updated_at BEFORE UPDATE ON works FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER episodes_updated_at BEFORE UPDATE ON episodes FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER lorebooks_updated_at BEFORE UPDATE ON lorebooks FOR EACH ROW EXECUTE FUNCTION update_updated_at();

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

-- ========================================
-- Views
-- ========================================
CREATE VIEW active_universes AS SELECT * FROM universes WHERE deleted_at IS NULL;
CREATE VIEW active_works AS SELECT * FROM works WHERE deleted_at IS NULL;
CREATE VIEW active_episodes AS SELECT * FROM episodes WHERE deleted_at IS NULL;
CREATE VIEW active_lorebooks AS SELECT * FROM lorebooks WHERE deleted_at IS NULL;

-- ========================================
-- Sample Data
-- ========================================
INSERT INTO universes (owner_id, title, description) VALUES
(1001, '룬테라 세계관', '마법과 룬이 지배하는 판타지 세계');

INSERT INTO works (universe_id, primary_author_id, title, synopsis, genre, status) VALUES
(1, 1001, '그웬의 여정', '살아있는 인형 그웬의 이야기', '다크 판타지', '연재중'),
(1, 1001, '비에고의 복수', '몰락한 왕 비에고의 이야기', '다크 판타지', '연재중');

INSERT INTO work_authors (work_id, user_id, role, contribution_percent) VALUES
(1, 1002, 'CO_AUTHOR', 20);

INSERT INTO episodes (work_id, user_id, ep_num, title, subtitle, txt_path, word_count, is_analyzed) VALUES
(1, 1001, 1, '첫 번째 이야기', '각성', '/storage/works/1/ep1.txt', 3500, true),
(1, 1001, 2, '두 번째 이야기', '기억', '/storage/works/1/ep2.txt', 4200, true),
(1, 1001, 3, '세 번째 이야기', '결의', '/storage/works/1/ep3.txt', 3800, false);

INSERT INTO lorebooks (universe_id, user_id, keyword, category, ep_num, setting) VALUES
(1, 1001, '그웬', '인물', ARRAY[1, 2], 
'{"name":"그웬","type":"인형","ability":["신성한 안개","마법 가위"]}'::jsonb),
(1, 1001, '비에고', '인물', ARRAY[1], 
'{"name":"비에고","type":"언데드","ability":["검은 안개","망령 소환"]}'::jsonb),
(1, 1001, '축복의 빛 군도', '장소', ARRAY[1, 2], 
'{"name":"축복의 빛 군도","type":"저주받은 땅","regions":["헬리아","녹시"]}'::jsonb);
