"""
定时任务调度器模块
用于管理系统的定期任务，如清理过期会话等
"""
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger


class TaskScheduler:
    """定时任务调度器单例类"""
    
    _instance = None
    _scheduler = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._scheduler is None:
            self._scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
    
    def start(self):
        """启动调度器"""
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("定时任务调度器已启动")
    
    def shutdown(self):
        """停止调度器"""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("定时任务调度器已停止")
    
    def add_cleanup_expired_task(self, short_term_memory):
        """
        添加清理过期会话的定时任务
        
        Args:
            short_term_memory: ShortTermMemory实例
        """
        async def cleanup_task():
            """清理过期会话的异步任务"""
            try:
                logger.info("开始执行清理过期会话任务...")
                deleted_count = await short_term_memory.cleanup_expired()
                logger.info(f"清理过期会话任务完成，删除了 {deleted_count} 条记录")
            except Exception as e:
                logger.error(f"清理过期会话任务执行失败：{e}")
        
        self._scheduler.add_job(
            cleanup_task,
            CronTrigger(hour=2, minute=0, timezone="Asia/Shanghai"),
            id="cleanup_expired_sessions",
            name="清理过期会话",
            replace_existing=True
        )
        logger.info("已添加定时任务：每天凌晨2点清理过期会话")

    def add_cleanup_old_memories_task(self, long_term_memory, user_ids: list = None, retention_days: int = 90):
        """
        添加清理长期记忆的定时任务（V2: 多级清理策略）

        清理策略:
        1. 删除低价值记忆(value_score < 0.4)
        2. 清理超过保留天数的旧记忆(仅低价值)
        3. 清理重复记忆
        4. 容量控制(超出限制时淘汰末位10%)

        Args:
            long_term_memory: LongTermMemory或LongTermMemoryV2实例
            user_ids: 需要清理的用户ID列表（None表示运行时动态获取）
            retention_days: 记忆保留天数，默认90天
        """
        def cleanup_task():
            """清理长期记忆的任务"""
            try:
                logger.info("开始执行清理长期记忆任务...")

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    async def run_cleanup():
                        actual_user_ids = user_ids
                        if actual_user_ids is None:
                            actual_user_ids = await long_term_memory.get_all_user_ids()

                        if actual_user_ids:
                            total_old = 0
                            total_dup = 0
                            total_low = 0
                            total_evicted = 0

                            for user_id in actual_user_ids:
                                # V2新增: 清理低价值记忆
                                if hasattr(long_term_memory, 'cleanup_low_value'):
                                    low_count = await long_term_memory.cleanup_low_value(user_id)
                                    total_low += low_count

                                # 清理旧记忆
                                old_count = await long_term_memory.cleanup_old_memories(user_id, retention_days)
                                total_old += old_count

                                # 清理重复记忆
                                dup_count = await long_term_memory.cleanup_duplicates(user_id)
                                total_dup += dup_count

                                # V2新增: 容量控制
                                if hasattr(long_term_memory, 'enforce_memory_limit'):
                                    evict_count = await long_term_memory.enforce_memory_limit(user_id)
                                    total_evicted += evict_count

                            logger.info(
                                f"长期记忆清理完成: 低价值{total_low}条, "
                                f"旧记忆{total_old}条, 重复{total_dup}条, "
                                f"容量淘汰{total_evicted}条"
                            )
                        else:
                            logger.info("没有找到有长期记忆的用户，跳过清理")

                    loop.run_until_complete(run_cleanup())
                finally:
                    loop.close()

            except Exception as e:
                logger.error(f"清理长期记忆任务执行失败：{e}")

        self._scheduler.add_job(
            cleanup_task,
            CronTrigger(hour=3, minute=30, timezone="Asia/Shanghai"),
            id="cleanup_old_memories",
            name="清理旧长期记忆",
            replace_existing=True
        )
        logger.info(f"已添加定时任务：每天凌晨3:30清理长期记忆(V2多级策略)")


def get_scheduler() -> TaskScheduler:
    """获取调度器单例实例"""
    return TaskScheduler()
