-- Active: 1776605236797@@127.0.0.1@5433@agent_rag
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS long_term_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(128) NOT NULL,
    session_id VARCHAR(128),
    content TEXT NOT NULL,
    embedding vector(1024), 
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
   
    CONSTRAINT content_not_empty CHECK (char_length(content) > 0) 
);

-- 向量索引：HNSW算法加速相似度搜索
CREATE INDEX idx_long_term_embedding ON long_term_memories 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 复合索引：用户+时间范围查询
CREATE INDEX idx_user_created ON long_term_memories(user_id, created_at DESC);



-- 1. 删除记忆表（连数据带结构一起删）
DROP TABLE IF EXISTS long_term_memories;

-- 2. 如果有残留索引也删掉（保险）
DROP INDEX IF EXISTS idx_long_term_memories_embedding;

