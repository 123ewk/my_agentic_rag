"""
长期记忆V2数据模型

改进点：
1. 四类型分类: user_profile / fact / experience / preference
2. 价值评分: value_score 用于筛选和淘汰
3. 摘要字段: summary 用于快速预筛选
4. 访问统计: access_count + last_accessed_at 用于时间衰减
5. 替代关系: supersedes_id 用于版本管理
"""
from sqlalchemy import Table, Column, String, Text, Integer, REAL, DateTime, MetaData
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from pgvector.sqlalchemy import Vector
from sqlalchemy import func

metadata = MetaData()
EMBEDDING_DIM = 1024

long_term_memories_v2 = Table(
    'long_term_memories_v2',
    metadata,
    Column('id', UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()),
    Column('user_id', String(128), nullable=False, index=True),
    Column('memory_type', String(32), nullable=False),
    Column('content', Text, nullable=False),
    Column('summary', String(256), nullable=False, server_default=''),
    Column('embedding', Vector(EMBEDDING_DIM), nullable=True),
    Column('value_score', REAL, nullable=False, server_default='0.5'),
    Column('metadata', JSONB, server_default='{}'),
    Column('source_session_ids', ARRAY(String(128)), server_default='{}'),
    Column('access_count', Integer, server_default='0'),
    Column('last_accessed_at', DateTime(timezone=True), nullable=True),
    Column('created_at', DateTime(timezone=True), server_default=func.now()),
    Column('updated_at', DateTime(timezone=True), server_default=func.now()),
    Column('supersedes_id', UUID(as_uuid=True), nullable=True),
)
