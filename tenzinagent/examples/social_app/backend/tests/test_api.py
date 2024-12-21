"""
Backend API tests with pytest and async testing
"""
import pytest
from httpx import AsyncClient
from fastapi import status

from app.core.security import create_access_token
from app.models.user import User
from app.main import app

@pytest.mark.asyncio
class TestAPI:
    async def test_create_user(self, async_client: AsyncClient, db_session):
        """Test user registration"""
        response = await async_client.post(
            "/api/v1/users",
            json={
                "email": "test@example.com",
                "password": "password123",
                "name": "Test User"
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == "test@example.com"
        assert "id" in data
        assert "password" not in data
        
        # Verify user in database
        user = await User.get(db_session, data["id"])
        assert user is not None
        assert user.email == "test@example.com"
        
    async def test_login(self, async_client: AsyncClient, test_user: User):
        """Test user login"""
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "password123"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        
    async def test_get_current_user(self, async_client: AsyncClient, test_user: User):
        """Test getting current user details"""
        token = create_access_token(test_user.id)
        
        response = await async_client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_user.id
        assert data["email"] == test_user.email
        
    async def test_create_post(self, async_client: AsyncClient, test_user: User):
        """Test creating a post"""
        token = create_access_token(test_user.id)
        
        response = await async_client.post(
            "/api/v1/posts",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "content": "Test post content"
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["content"] == "Test post content"
        assert data["author"]["id"] == test_user.id
        
    async def test_like_post(self, async_client: AsyncClient, test_user: User, test_post):
        """Test liking a post"""
        token = create_access_token(test_user.id)
        
        response = await async_client.post(
            f"/api/v1/posts/{test_post.id}/like",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["likes_count"] == 1
        assert test_user.id in data["liked_by"]
        
    async def test_get_feed(self, async_client: AsyncClient, test_user: User):
        """Test getting user feed"""
        token = create_access_token(test_user.id)
        
        # Create some test posts
        for i in range(3):
            await async_client.post(
                "/api/v1/posts",
                headers={"Authorization": f"Bearer {token}"},
                json={"content": f"Test post {i}"}
            )
            
        response = await async_client.get(
            "/api/v1/feed",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["posts"]) == 3
        assert data["total"] == 3
        
        # Test pagination
        response = await async_client.get(
            "/api/v1/feed?limit=2&offset=1",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["posts"]) == 2
        
    @pytest.mark.parametrize(
        "query,expected_count",
        [
            ("test", 3),
            ("post 1", 1),
            ("nonexistent", 0)
        ]
    )
    async def test_search_posts(
        self,
        async_client: AsyncClient,
        test_user: User,
        query: str,
        expected_count: int
    ):
        """Test searching posts"""
        token = create_access_token(test_user.id)
        
        # Create test posts
        for i in range(3):
            await async_client.post(
                "/api/v1/posts",
                headers={"Authorization": f"Bearer {token}"},
                json={"content": f"Test post {i}"}
            )
            
        response = await async_client.get(
            f"/api/v1/posts/search?q={query}",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["posts"]) == expected_count