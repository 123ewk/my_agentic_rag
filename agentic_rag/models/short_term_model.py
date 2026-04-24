# SQLAlchemy表定义
from sqlalchemy import Table, Column, String, Integer, Text, DateTime, MetaData
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import func

# 1. 先定义文件级metadata（规范写法，避免爆红） # 创建元数据容器 
# 每个 MetaData() 实例是独立的,只有通过同一个 metadata 对象创建的表才会被收集在一起
metadata = MetaData()

# 2. 正确定义表，去掉错误的Column[类型]标注
conversation_sessions = Table(
    'conversation_sessions',
    metadata,  # 绑定metadata，不再用None
    Column('id', UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()),
    Column('session_id', String(128), nullable=False, index=True),
    Column('message_index', Integer, nullable=False),
    Column('role', String(16), nullable=False),
    Column('content', Text, nullable=False),
    Column('metadata', JSONB, server_default='{}'),
    Column('token_count', Integer, server_default='0'),
    Column('created_at', DateTime(timezone=True), server_default=func.now()),
    Column('expires_at', DateTime(timezone=True), nullable=True),
)
