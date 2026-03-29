
import jwt
import pytest
import asyncio
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from app.main import app
from app.config.settings import get_settings

settings = get_settings()

def generate_token(user_id="123", role="student", username="testuser"):
    payload = {
        "user_id": user_id,
        "role": role,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

from contextlib import asynccontextmanager

@asynccontextmanager
async def mock_lifespan(app):
    yield {}

def test_rate_limiting():
    user_id = "test_user_rate_limit"
    token = generate_token(user_id=user_id)
    headers = {"Authorization": f"Bearer {token}"}
    
    # Mock everything that happens during lifespan and dependency injection
    with patch("app.main.lifespan", side_effect=mock_lifespan), \
         patch("app.database.postgres.init_db", new_callable=AsyncMock), \
         patch("app.database.redis_client.get_redis_client") as mock_get_redis_lifespan, \
         patch("app.database.qdrant_client.get_qdrant_client") as mock_get_qdrant, \
         patch("app.utils.logger_batch.batch_logger.start", new_callable=AsyncMock), \
         patch("app.api.auth.get_redis_client") as mock_get_redis:
        
        # Setup mock redis for the lifespan
        mock_redis_lifespan = AsyncMock()
        mock_get_redis_lifespan.return_value = mock_redis_lifespan
        
        # Setup mock qdrant
        mock_qdrant = AsyncMock()
        mock_get_qdrant.return_value = mock_qdrant
        
        # Setup mock redis for the auth dependency
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis
        
        # Simulate request counts: 1, 2, 3, 4, 5, 6
        mock_redis.incr.side_effect = [1, 2, 3, 4, 5, 6]
        mock_redis.expire.return_value = True
        
        with patch("app.api.routes.chat.resolve_numeric_query", new_callable=AsyncMock) as mock_resolve, \
             patch("app.api.routes.chat.get_cached_response", new_callable=AsyncMock) as mock_cache, \
             patch("app.api.routes.chat.get_conversation_history", new_callable=AsyncMock) as mock_history, \
             patch("app.api.routes.chat.get_rag_graph") as mock_graph:
            
            mock_resolve.side_effect = lambda q, cid: q
            mock_cache.return_value = None
            mock_history.return_value = []
            
            # Mock graph.ainvoke
            mock_graph_instance = AsyncMock()
            mock_graph_instance.ainvoke.return_value = {
                "messages": [MagicMock(content="Hello! I am an AI.")],
                "retrieved_context": []
            }
            mock_graph.return_value = mock_graph_instance
            
            # Disable lifespan to avoid connection errors entirely
            app.router.lifespan_context = mock_lifespan
            
            with TestClient(app) as client:
                # Make 5 requests (should succeed)
                for i in range(5):
                    response = client.post(
                        "/api/v1/chat",
                        json={"query": f"request {i}", "course_id": 1},
                        headers=headers
                    )
                    assert response.status_code == 200, f"Request {i} failed: {response.text}"
                
                # Make the 6th request (should fail with 429)
                response = client.post(
                    "/api/v1/chat",
                    json={"query": "request 6", "course_id": 1},
                    headers=headers
                )
                print(f"6th request status: {response.status_code}")
                print(f"6th request body: {response.text}")
                assert response.status_code == 429
                assert response.json()["detail"] == "Rate limit exceeded. Please try again later."
            
            # Verify Redis calls
            assert mock_redis.incr.call_count == 6
            mock_redis.expire.assert_awaited_with(f"rate_limit:{user_id}", 60)

if __name__ == "__main__":
    test_rate_limiting()
