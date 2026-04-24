"""
测试记忆功能脚本
"""
import asyncio
from agentic_rag.memory.short_term import ShortTermMemory
from agentic_rag.memory.long_term import LongTermMemory
from agentic_rag.vectorstore.embeddings import get_embeddings
from agentic_rag.config.settings import get_settings
from loguru import logger

async def test_short_term_memory():
    """测试短期记忆"""
    settings = get_settings()
    
    logger.info("=" * 50)
    logger.info("测试短期记忆功能")
    logger.info("=" * 50)
    
    # 初始化短期记忆
    memory = ShortTermMemory(
        database_url=settings.database_url,
        max_messages=10,
        max_tokens=5000,
        ttl_hours=24
    )
    
    # 连接数据库
    await memory.connect()
    logger.info("✅ 短期记忆连接成功")
    
    # 测试会话ID
    test_session_id = "test_session_001"
    
    # 1. 添加测试消息
    logger.info("\n📝 步骤1: 添加测试消息")
    await memory.add_message(
        session_id=test_session_id,
        question="我叫张三，我喜欢编程",
        answer="你好张三！我记住了你的名字和兴趣。"
    )
    logger.info("✅ 消息添加成功")
    
    # 2. 获取消息历史
    logger.info("\n📖 步骤2: 获取消息历史")
    messages = await memory.get_message(test_session_id)
    logger.info(f"获取到 {len(messages)} 条消息")
    for i, msg in enumerate(messages):
        role = "用户" if hasattr(msg, 'type') and msg.type == 'human' else "助手"
        logger.info(f"  [{i+1}] {role}: {msg.content}")
    
    # 3. 测试上下文获取
    logger.info("\n📋 步骤3: 获取上下文")
    context = await memory.get_context(test_session_id)
    logger.info(f"上下文内容:\n{context}")
    
    # 4. 清理测试数据
    logger.info("\n🧹 步骤4: 清理测试数据")
    deleted = await memory.clear_session(test_session_id)
    logger.info(f"删除了 {deleted} 条消息")
    
    # 关闭连接
    await memory.close()
    logger.info("✅ 测试完成")
    
    return True

async def test_long_term_memory():
    """测试长期记忆"""
    settings = get_settings()
    
    logger.info("\n" + "=" * 50)
    logger.info("测试长期记忆功能")
    logger.info("=" * 50)
    
    # 初始化嵌入模型
    embeddings = get_embeddings(
        embedding_type="zhipu",
        api_key=settings.zhipu_api_key,
        model_name=settings.embedding_model_name,
        dimension=settings.embedding_dimension
    )
    logger.info("✅ 嵌入模型初始化成功")
    
    # 初始化长期记忆
    memory = LongTermMemory(
        embeddings=embeddings,
        database_url=settings.database_url,
        k=5,
        similarity_threshold=0.7
    )
    
    # 连接数据库
    await memory.connect()
    logger.info("✅ 长期记忆连接成功")
    
    # 测试用户ID
    test_user_id = "test_user_001"
    
    # 1. 保存记忆
    logger.info("\n📝 步骤1: 保存记忆")
    memory_id = await memory.save_memory(
        user_id=test_user_id,
        content="用户张三喜欢使用Python进行数据分析",
        metadata={"topic": "user_preference", "language": "python"}
    )
    logger.info(f"✅ 记忆保存成功, ID: {memory_id}")
    
    # 2. 搜索记忆
    logger.info("\n🔍 步骤2: 搜索记忆")
    results = await memory.search(test_user_id, "张三喜欢什么编程语言")
    logger.info(f"找到 {len(results)} 条相关记忆")
    for i, result in enumerate(results):
        logger.info(f"  [{i+1}] 相似度: {result['similarity']:.3f}")
        logger.info(f"      内容: {result['content']}")
    
    # 3. 获取统计
    logger.info("\n📊 步骤3: 获取统计信息")
    stats = await memory.get_stats(test_user_id)
    logger.info(f"用户 {test_user_id} 的记忆统计: {stats}")
    
    # 4. 删除记忆
    logger.info("\n🧹 步骤4: 删除测试记忆")
    deleted = await memory.delete_memory(memory_id, test_user_id)
    logger.info(f"✅ 记忆删除{'成功' if deleted else '失败'}")
    
    # 关闭连接
    await memory.close()
    logger.info("✅ 测试完成")
    
    return True

async def main():
    """主测试函数"""
    logger.info("\n" + "=" * 60)
    logger.info("开始测试记忆功能")
    logger.info("=" * 60)
    
    try:
        # 测试短期记忆
        await test_short_term_memory()
        
        # 测试长期记忆
        await test_long_term_memory()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 所有测试通过！")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
