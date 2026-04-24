"""
数据库表初始化模块
确保短期记忆和长期记忆所需的数据库表已创建
"""
import asyncio
from sqlalchemy import text
from loguru import logger


async def init_memory_tables(database_url: str):
    """
    初始化记忆相关的数据库表
    
    Args:
        database_url: PostgreSQL连接字符串
    """
    from sqlalchemy.ext.asyncio import create_async_engine
    
    engine = create_async_engine(database_url, echo=False)
    
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            await conn.execute(text("""
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
                    CONSTRAINT unique_session_message UNIQUE (session_id, message_index)
                )
            """))
            
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(128) NOT NULL,
                    session_id VARCHAR(128),
                    content TEXT NOT NULL,
                    embedding vector(1024),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    CONSTRAINT content_not_empty CHECK (char_length(content) > 0)
                )
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_conversation_session_id 
                ON conversation_sessions(session_id)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_conversation_expires 
                ON conversation_sessions(expires_at) WHERE expires_at IS NOT NULL
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_long_term_user_id 
                ON long_term_memories(user_id)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_long_term_embedding 
                ON long_term_memories 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            
        logger.info("✅ 数据库表初始化完成")
        
    except Exception as e:
        logger.error(f"数据库表初始化失败: {e}")
        raise
    finally:
        await engine.dispose()


def init_memory_tables_sync(database_url: str):
    """
    同步版本的表初始化（供启动时调用）
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(init_memory_tables(database_url))
    finally:
        loop.close()