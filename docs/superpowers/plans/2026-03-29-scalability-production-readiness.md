# Scalability and Production-Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the AI LMS Agent into a production-ready system capable of handling 600 DAU and 12,000 total users with robust background processing, batched logging, and secure endpoints.

**Architecture:** 
1. **Resource Optimization**: Scale connection pools for PostgreSQL and Redis.
2. **Asynchronous Processing**: Introduce `arq` for persistent background tasks (Moodle Sync).
3. **High-Performance Logging**: Implement a Redis-buffered batch logger for chat interactions.
4. **Security**: Add JWT-based authentication and Redis-backed rate limiting.

**Tech Stack:** FastAPI, PostgreSQL (SQLAlchemy), Redis (aioredis), Arq (background tasks), PyJWT.

---

### Task 1: Connection Pooling & Resource Scaling

**Files:**
- Modify: `app/database/postgres.py`
- Modify: `app/database/redis_client.py`
- Modify: `app/config/settings.py`

- [ ] **Step 1: Update settings to include pool configuration**
  Add `postgres_pool_size`, `postgres_max_overflow`, and `redis_max_connections` to `Settings` class in `app/config/settings.py`.

- [ ] **Step 2: Apply pooling to PostgreSQL**
  ```python
  # app/database/postgres.py
  engine = create_async_engine(
      settings.postgres_dsn,
      pool_size=settings.postgres_pool_size, # default 20
      max_overflow=settings.postgres_max_overflow, # default 40
      pool_pre_ping=True,
  )
  ```

- [ ] **Step 3: Apply pooling to Redis**
  ```python
  # app/database/redis_client.py
  def _create_pool() -> aioredis.ConnectionPool:
      return aioredis.ConnectionPool.from_url(
          settings.redis_url,
          max_connections=settings.redis_max_connections, # default 100
          # ... existing ...
      )
  ```

- [ ] **Step 4: Commit scaling changes**
  `git commit -m "feat: scale database connection pools for high concurrency"`

---

### Task 2: Persistent Background Workers (Moodle Sync)

**Files:**
- Create: `app/worker.py`
- Modify: `app/api/routes/ingest.py`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Install `arq`**
  Run: `pip install arq` and update `pyproject.toml`.

- [ ] **Step 2: Create Worker Definition**
  Create `app/worker.py` to define the `sync_moodle_task`. This worker will initialize its own DB and Qdrant connections.

- [ ] **Step 3: Refactor Ingest Route to use Arq**
  Modify `app/api/routes/ingest.py` to use `arq.create_pool()` and `redis.enqueue_job('sync_moodle_task', ...)` instead of `background_tasks.add_task`.

- [ ] **Step 4: Update Docker Compose**
  Add a `worker` service that runs `arq app.worker.WorkerSettings`.

- [ ] **Step 5: Commit background worker**
  `git commit -m "feat: migrate moodle sync to persistent arq worker"`

---

### Task 3: Batched Database Logging

**Files:**
- Create: `app/utils/logger_batch.py`
- Modify: `app/api/routes/chat.py`
- Modify: `app/main.py`

- [ ] **Step 1: Create Batch Logger Utility**
  Implement a class that buffers `AgentLog` entries in a Redis list and flushes them to PostgreSQL in bulk (e.g., every 50 logs or 10 seconds).

- [ ] **Step 2: Update Chat Route**
  Replace individual `log_to_db` calls in `app/api/routes/chat.py` with `batch_logger.add_log(log_entry)`.

- [ ] **Step 3: Initialize/Shutdown Batch Logger**
  Add lifecycle hooks in `app/main.py` to start the batch flushing task and ensure final flush on shutdown.

- [ ] **Step 4: Commit batch logging**
  `git commit -m "perf: implement batched logging for chat interactions"`

---

### Task 4: JWT Authentication Middleware

**Files:**
- Create: `app/api/auth.py`
- Modify: `app/main.py`
- Modify: `app/api/routes/chat.py`
- Modify: `app/api/routes/ingest.py`

- [ ] **Step 1: Create Auth Dependency**
  Implement `get_current_user` in `app/api/auth.py` that validates a JWT token from the `Authorization` header.

- [ ] **Step 2: Apply Security to Routes**
  Add `Depends(get_current_user)` to all sensitive routes.

- [ ] **Step 3: Commit authentication**
  `git commit -m "security: add JWT-based authentication middleware"`

---

### Task 5: Redis Rate Limiting

**Files:**
- Modify: `app/api/auth.py` (or new middleware)
- Modify: `app/api/routes/chat.py`

- [ ] **Step 1: Implement Rate Limiter**
  Use Redis `INCR` with `EXPIRE` to limit chat requests to e.g., 5 per minute per user.

- [ ] **Step 2: Commit rate limiting**
  `git commit -m "security: implement redis-backed rate limiting per user"`
