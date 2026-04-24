-- Active: 1776605236797@@127.0.0.1@5433@agent_rag
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(128) NOT NULL, 
    message_index INTEGER NOT NULL, 
    role VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL, 
    metadata JSONB DEFAULT '{}', 
    token_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    
    # 唯一约束：会话ID+消息索引组合,防重复数据
    CONSTRAINT unique_session_message UNIQUE (session_id, message_index)
);


-- idx_session_id：按会话ID查询消息，优化会话历史检索
CREATE INDEX idx_session_id ON conversation_sessions(session_id);

-- idx_expires_at：按过期时间查询过期会话，优化会话清理: 创建部分索引(性能优化的高级用法): 只给有过期时间的记录建索引，NULL 值不索引。
CREATE INDEX idx_expires_at ON conversation_sessions(expires_at) WHERE expires_at IS NOT NULL;

-- 自动清理过期会话的定时任务（每天凌晨2点执行）
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    DELETE FROM conversation_sessions 
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
END;
$$ LANGUAGE plpgsql;


# 查看定时任务
