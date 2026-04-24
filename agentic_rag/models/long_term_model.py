# SQLAlchemy表定义
from sqlalchemy import Table, Column, String, Text, DateTime, MetaData
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from sqlalchemy import func

metadata = MetaData()
# 向量模型维度为1024
EMBEDDING_DIM = 1024

long_term_memories = Table(
    'long_term_memories',
    metadata,
    Column('id', UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()),
    Column('user_id', String(128), nullable=False, index=True),
    Column('session_id', String(128), nullable=True, index=True),
    Column('content', Text, nullable=False),
    Column('embedding', Vector(EMBEDDING_DIM), nullable=True),
    Column('metadata', JSONB, server_default='{}'),
    Column('created_at', DateTime(timezone=True), server_default=func.now()),
)