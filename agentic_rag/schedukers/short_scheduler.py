"""
定时任务调度器模块
用于管理系统的定期任务，如清理过期会话等
"""
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
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
            self._scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
    
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
        
        # 每天凌晨2点执行清理任务
        self._scheduler.add_job(
            cleanup_task,
            CronTrigger(hour=2, minute=0, timezone="Asia/Shanghai"),
            id="cleanup_expired_sessions",
            name="清理过期会话",
            replace_existing=True
        )
        logger.info("已添加定时任务：每天凌晨2点清理过期会话")


def get_scheduler() -> TaskScheduler:
    """获取调度器单例实例"""
    return TaskScheduler()
