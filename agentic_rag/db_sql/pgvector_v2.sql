-- 长期记忆V2表结构
-- 改进: 四类型分类 + 价值评分 + 摘要 + 访问统计 + 替代关系

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS long_term_memories_v2 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(128) NOT NULL,
    memory_type VARCHAR(32) NOT NULL,
    content TEXT NOT NULL,
    summary VARCHAR(256) NOT NULL DEFAULT '',
    embedding vector(1024),
    value_score REAL NOT NULL DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',
    source_session_ids VARCHAR(128)[] DEFAULT '{}',
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    supersedes_id UUID,

    CONSTRAINT content_not_empty CHECK (char_length(content) > 0),
    CONSTRAINT valid_memory_type CHECK (memory_type IN ('user_profile', 'fact', 'experience', 'preference')),
    CONSTRAINT valid_value_score CHECK (value_score >= 0 AND value_score <= 1)
);

-- 按类型+用户复合索引
CREATE INDEX IF NOT EXISTS idx_v2_memory_type_user ON long_term_memories_v2(user_id, memory_type);

-- 价值分数索引(低价值记忆定期清理)
CREATE INDEX IF NOT EXISTS idx_v2_value_score ON long_term_memories_v2(user_id, value_score);

-- HNSW向量索引
CREATE INDEX IF NOT EXISTS idx_v2_embedding ON long_term_memories_v2
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 访问统计索引(时间衰减计算)
CREATE INDEX IF NOT EXISTS idx_v2_last_accessed ON long_term_memories_v2(user_id, last_accessed_at DESC);

-- 用户+创建时间索引
CREATE INDEX IF NOT EXISTS idx_v2_user_created ON long_term_memories_v2(user_id, created_at DESC);
