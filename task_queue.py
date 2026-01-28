"""
Simple in-memory task queue for asynchronous invoice processing.
This allows invoices to be processed in the background while returning
immediate responses to users.
"""
import asyncio
import uuid
from typing import Dict, Optional, Callable, Any
from enum import Enum
from datetime import datetime
import logging

log = logging.getLogger("task-queue")

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskQueue:
    """
    In-memory task queue with background workers.
    
    Usage:
        queue = TaskQueue(max_workers=3)
        await queue.start_workers()
        
        job_id = await queue.enqueue(process_function, arg1, arg2, kwarg1=value1)
        status = queue.get_job(job_id)
    """
    
    def __init__(self, max_workers: int = 3):
        self.queue = asyncio.Queue()
        self.jobs: Dict[str, Dict] = {}
        self.max_workers = max_workers
        self.workers = []
        self.running = False
    
    async def start_workers(self):
        """Start background worker tasks"""
        if self.running:
            return
        
        self.running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        log.info(f"Started {self.max_workers} background workers")
    
    async def stop_workers(self):
        """Stop background workers"""
        self.running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers = []
        log.info("Stopped all background workers")
    
    async def _worker(self, name: str):
        """Worker that processes tasks from queue"""
        log.info(f"Worker {name} started")
        while self.running:
            try:
                # Wait for task (with timeout to check if still running)
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                job_id, func, args, kwargs = task
                
                # Update job status
                self.jobs[job_id]["status"] = JobStatus.PROCESSING.value
                self.jobs[job_id]["started_at"] = datetime.now().isoformat()
                
                log.info(f"Worker {name} processing job {job_id}")
                
                # Execute task
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, func, *args, **kwargs)
                    
                    # Update job with result
                    self.jobs[job_id]["status"] = JobStatus.COMPLETED.value
                    self.jobs[job_id]["result"] = result
                    self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
                    log.info(f"Job {job_id} completed successfully")
                    
                except Exception as e:
                    log.exception(f"Job {job_id} failed: {e}")
                    self.jobs[job_id]["status"] = JobStatus.FAILED.value
                    self.jobs[job_id]["error"] = str(e)
                    self.jobs[job_id]["failed_at"] = datetime.now().isoformat()
                
                finally:
                    self.queue.task_done()
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.exception(f"Worker {name} error: {e}")
    
    async def enqueue(self, func: Callable, *args, **kwargs) -> str:
        """
        Add task to queue and return job ID.
        
        Args:
            func: Function to execute (can be sync or async)
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            job_id: Unique identifier for the job
        """
        job_id = str(uuid.uuid4())
        
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING.value,
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None
        }
        
        await self.queue.put((job_id, func, args, kwargs))
        log.info(f"Job {job_id} enqueued")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status and result"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> Dict[str, Dict]:
        """Get all jobs (for debugging/monitoring)"""
        return self.jobs
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """
        Remove completed/failed jobs older than max_age_hours.
        This prevents memory buildup over time.
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        jobs_to_remove = []
        for job_id, job_data in self.jobs.items():
            if job_data["status"] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                completed_at = job_data.get("completed_at") or job_data.get("failed_at")
                if completed_at:
                    try:
                        job_time = datetime.fromisoformat(completed_at)
                        if job_time < cutoff:
                            jobs_to_remove.append(job_id)
                    except Exception:
                        pass
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        if jobs_to_remove:
            log.info(f"Cleaned up {len(jobs_to_remove)} old jobs")

# Global task queue instance
task_queue = TaskQueue(max_workers=3)










