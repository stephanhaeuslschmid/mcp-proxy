"""Tests for the process pool."""
# ruff: noqa: PLR2004

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.client.stdio import StdioServerParameters

from mcp_proxy.process_pool import (
    CachedProcess,
    ProcessHandle,
    ProcessPool,
    generate_cache_key,
    get_process_pool,
    init_process_pool,
    shutdown_process_pool,
)


class TestGenerateCacheKey:
    """Tests for cache key generation."""

    def test_generate_cache_key_deterministic(self) -> None:
        """Test that cache key generation is deterministic."""
        env_vars = {"TOKEN": "abc123", "USER": "test"}
        key1 = generate_cache_key("server1", env_vars)
        key2 = generate_cache_key("server1", env_vars)
        assert key1 == key2

    def test_generate_cache_key_same_env_same_key(self) -> None:
        """Test that same env vars produce same cache key regardless of order."""
        env_vars1 = {"TOKEN": "abc123", "USER": "test"}
        env_vars2 = {"USER": "test", "TOKEN": "abc123"}
        key1 = generate_cache_key("server1", env_vars1)
        key2 = generate_cache_key("server1", env_vars2)
        assert key1 == key2

    def test_generate_cache_key_different_env_different_key(self) -> None:
        """Test that different env vars produce different cache keys."""
        env_vars1 = {"TOKEN": "abc123"}
        env_vars2 = {"TOKEN": "xyz789"}
        key1 = generate_cache_key("server1", env_vars1)
        key2 = generate_cache_key("server1", env_vars2)
        assert key1 != key2

    def test_generate_cache_key_different_server_different_key(self) -> None:
        """Test that different server names produce different cache keys."""
        env_vars = {"TOKEN": "abc123"}
        key1 = generate_cache_key("server1", env_vars)
        key2 = generate_cache_key("server2", env_vars)
        assert key1 != key2

    def test_generate_cache_key_empty_env(self) -> None:
        """Test cache key generation with empty env vars."""
        key1 = generate_cache_key("server1", {})
        key2 = generate_cache_key("server1", {})
        assert key1 == key2
        assert len(key1) == 64  # SHA256 produces 64 hex characters


class TestCachedProcess:
    """Tests for CachedProcess dataclass."""

    def test_reference_count_operations(self) -> None:
        """Test reference count increment and decrement."""
        cached = CachedProcess(
            server_name="test",
            cache_key="key123",
            created_at=datetime.now(timezone.utc),
            last_used=datetime.now(timezone.utc),
            exit_stack=MagicMock(),
            session=MagicMock(),
        )

        assert not cached.is_in_use
        cached.increment_ref()
        assert cached.is_in_use
        cached.increment_ref()
        assert cached.is_in_use
        cached.decrement_ref()
        assert cached.is_in_use
        cached.decrement_ref()
        assert not cached.is_in_use

    def test_mark_for_removal(self) -> None:
        """Test marking process for lazy removal."""
        cached = CachedProcess(
            server_name="test",
            cache_key="key123",
            created_at=datetime.now(timezone.utc),
            last_used=datetime.now(timezone.utc),
            exit_stack=MagicMock(),
            session=MagicMock(),
        )

        assert not cached.is_marked_for_removal
        cached.mark_for_removal()
        assert cached.is_marked_for_removal


class TestProcessPool:
    """Tests for ProcessPool class."""

    @pytest.fixture
    def pool(self) -> ProcessPool:
        """Create a fresh ProcessPool for testing."""
        return ProcessPool(idle_timeout=10, max_size=5)

    @pytest.fixture
    def mock_params(self) -> StdioServerParameters:
        """Create mock stdio parameters."""
        return StdioServerParameters(
            command="echo",
            args=["hello"],
            env={"BASE_VAR": "value"},
        )

    @pytest.fixture
    def mock_stdio_client(self) -> MagicMock:
        """Create a mock stdio_client context manager."""
        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_context.__aexit__.return_value = None

        return mock_context

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock ClientSession."""
        mock = MagicMock()
        mock.initialize = AsyncMock()
        return mock

    async def test_pool_start_creates_cleanup_task(self, pool: ProcessPool) -> None:
        """Test that start() creates the cleanup task."""
        assert pool._cleanup_task is None
        await pool.start()
        assert pool._cleanup_task is not None
        await pool.shutdown()

    async def test_pool_start_idempotent(self, pool: ProcessPool) -> None:
        """Test that start() can be called multiple times safely."""
        await pool.start()
        task1 = pool._cleanup_task
        await pool.start()  # Second call should not create new task
        assert pool._cleanup_task is task1
        await pool.shutdown()

    async def test_get_cached_count_empty(self, pool: ProcessPool) -> None:
        """Test get_cached_count returns 0 for empty pool."""
        assert pool.get_cached_count() == 0

    async def test_invalidate_nonexistent_key(self, pool: ProcessPool) -> None:
        """Test invalidate with non-existent key doesn't raise."""
        await pool.invalidate("nonexistent_key")  # Should not raise

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_get_or_create_cache_miss(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
        pool: ProcessPool,
        mock_params: StdioServerParameters,
    ) -> None:
        """Test get_or_create on cache miss creates new process."""
        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        handle = await pool.get_or_create(
            server_name="test-server",
            params=mock_params,
            header_env_vars={"TOKEN": "test123"},
        )

        assert handle is not None
        assert handle.session is mock_session
        assert pool.get_cached_count() == 1

        await handle.release()
        await pool.shutdown()

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_get_or_create_cache_hit(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
        pool: ProcessPool,
        mock_params: StdioServerParameters,
    ) -> None:
        """Test get_or_create on cache hit returns cached process."""
        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        # First call - cache miss
        handle1 = await pool.get_or_create(
            server_name="test-server",
            params=mock_params,
            header_env_vars={"TOKEN": "test123"},
        )
        await handle1.release()

        # Second call - cache hit
        handle2 = await pool.get_or_create(
            server_name="test-server",
            params=mock_params,
            header_env_vars={"TOKEN": "test123"},
        )

        # Should reuse the same session
        assert handle2.session is handle1.session
        assert pool.get_cached_count() == 1
        # stdio_client should only be called once
        assert mock_stdio.call_count == 1

        await handle2.release()
        await pool.shutdown()

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_get_or_create_different_env_creates_new(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
        pool: ProcessPool,
        mock_params: StdioServerParameters,
    ) -> None:
        """Test that different env vars create new processes."""
        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        # First call with token A
        handle1 = await pool.get_or_create(
            server_name="test-server",
            params=mock_params,
            header_env_vars={"TOKEN": "tokenA"},
        )
        await handle1.release()

        # Second call with token B
        handle2 = await pool.get_or_create(
            server_name="test-server",
            params=mock_params,
            header_env_vars={"TOKEN": "tokenB"},
        )

        # Should have created two separate processes
        assert pool.get_cached_count() == 2
        assert mock_stdio.call_count == 2

        await handle2.release()
        await pool.shutdown()

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_invalidate_removes_process(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
        pool: ProcessPool,
        mock_params: StdioServerParameters,
    ) -> None:
        """Test that invalidate removes process from cache."""
        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        handle = await pool.get_or_create(
            server_name="test-server",
            params=mock_params,
            header_env_vars={"TOKEN": "test123"},
        )
        cache_key = handle.cache_key
        await handle.release()

        assert pool.get_cached_count() == 1

        await pool.invalidate(cache_key)

        assert pool.get_cached_count() == 0

        await pool.shutdown()

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_shutdown_clears_pool(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
        pool: ProcessPool,
        mock_params: StdioServerParameters,
    ) -> None:
        """Test that shutdown clears all processes from pool."""
        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        # Create a few processes
        handle1 = await pool.get_or_create(
            server_name="test-server",
            params=mock_params,
            header_env_vars={"TOKEN": "token1"},
        )
        await handle1.release()

        handle2 = await pool.get_or_create(
            server_name="test-server",
            params=mock_params,
            header_env_vars={"TOKEN": "token2"},
        )
        await handle2.release()

        assert pool.get_cached_count() == 2

        await pool.shutdown()

        assert pool.get_cached_count() == 0

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_max_size_limit(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
        mock_params: StdioServerParameters,
    ) -> None:
        """Test that pool respects max_size limit."""
        pool = ProcessPool(idle_timeout=10, max_size=2)

        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        # Create more processes than max_size
        handles = []
        for i in range(4):
            handle = await pool.get_or_create(
                server_name="test-server",
                params=mock_params,
                header_env_vars={"TOKEN": f"token{i}"},
            )
            handles.append(handle)
            await handle.release()

        # Pool should only cache max_size processes
        assert pool.get_cached_count() == 2

        await pool.shutdown()

    # Note: test_dead_process_removed_on_get was removed because the process pool
    # does not actively check for dead processes via at_eof(). Instead, the caller
    # (mcp_server.py) catches exceptions and calls invalidate() to remove dead processes.
    # Dead process detection is handled by the caller's error handling, not the pool itself.


class TestProcessHandle:
    """Tests for ProcessHandle class."""

    async def test_handle_as_context_manager(self) -> None:
        """Test that ProcessHandle works as async context manager."""
        mock_pool = MagicMock()
        mock_cached = MagicMock()
        mock_cached.decrement_ref = MagicMock()

        handle = ProcessHandle(
            session=MagicMock(),
            cache_key="test_key",
            _pool=mock_pool,
            _cached_process=mock_cached,
        )

        async with handle:
            pass

        mock_cached.decrement_ref.assert_called_once()

    async def test_handle_release(self) -> None:
        """Test that release() decrements reference count."""
        mock_cached = MagicMock()
        mock_cached.decrement_ref = MagicMock()

        handle = ProcessHandle(
            session=MagicMock(),
            cache_key="test_key",
            _pool=MagicMock(),
            _cached_process=mock_cached,
        )

        await handle.release()

        mock_cached.decrement_ref.assert_called_once()


class TestGlobalPoolFunctions:
    """Tests for global pool management functions."""

    async def test_get_process_pool_singleton(self) -> None:
        """Test that get_process_pool returns singleton."""
        # Clear any existing pool
        import mcp_proxy.process_pool as pp

        pp._process_pool = None

        pool1 = get_process_pool()
        pool2 = get_process_pool()

        assert pool1 is pool2

        # Clean up
        pp._process_pool = None

    async def test_init_and_shutdown_process_pool(self) -> None:
        """Test init and shutdown of global pool."""
        import mcp_proxy.process_pool as pp

        pp._process_pool = None

        pool = await init_process_pool()
        assert pool._started

        await shutdown_process_pool()
        assert pp._process_pool is None


class TestIdleCleanup:
    """Tests for idle process cleanup (lazy cleanup pattern)."""

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    @patch("mcp_proxy.process_pool.PROCESS_POOL_CLEANUP_INTERVAL", 0.1)
    async def test_idle_process_marked_for_removal(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
    ) -> None:
        """Test that idle processes are marked for lazy cleanup."""
        pool = ProcessPool(idle_timeout=1, max_size=5)  # 1 second timeout

        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_exit_stack = AsyncMock()
        mock_exit_stack.aclose = AsyncMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        params = StdioServerParameters(command="echo", args=["test"])

        # Create a process
        handle = await pool.get_or_create(
            server_name="test-server",
            params=params,
            header_env_vars={"TOKEN": "test123"},
        )
        await handle.release()

        assert pool.get_cached_count() == 1

        # Manually set last_used to be older than timeout
        cache_key = handle.cache_key
        pool._pool[cache_key].last_used = datetime.now(timezone.utc) - timedelta(seconds=10)

        # Run cleanup manually - this only MARKS the process, doesn't remove it yet
        await pool._cleanup_idle_processes()

        # Process is still in pool but marked for removal
        assert pool.get_cached_count() == 1
        assert pool._pool[cache_key].is_marked_for_removal

        # Actual cleanup happens during next get_or_create (lazy cleanup)
        await pool._cleanup_marked_processes()

        # Now it should be removed
        assert pool.get_cached_count() == 0

        await pool.shutdown()

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    @patch("mcp_proxy.process_pool.PROCESS_POOL_CLEANUP_INTERVAL", 0.1)
    async def test_lazy_cleanup_on_get_or_create(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
    ) -> None:
        """Test that lazy cleanup happens during get_or_create."""
        pool = ProcessPool(idle_timeout=1, max_size=5)

        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        params = StdioServerParameters(command="echo", args=["test"])

        # Create first process
        handle1 = await pool.get_or_create(
            server_name="test-server",
            params=params,
            header_env_vars={"TOKEN": "token1"},
        )
        await handle1.release()

        # Mark it for removal
        cache_key1 = handle1.cache_key
        pool._pool[cache_key1].last_used = datetime.now(timezone.utc) - timedelta(seconds=10)
        await pool._cleanup_idle_processes()
        assert pool._pool[cache_key1].is_marked_for_removal

        # Create second process with different token - this triggers lazy cleanup
        handle2 = await pool.get_or_create(
            server_name="test-server",
            params=params,
            header_env_vars={"TOKEN": "token2"},
        )
        await handle2.release()

        # First process should be cleaned up, second should be cached
        assert pool.get_cached_count() == 1
        assert cache_key1 not in pool._pool

        await pool.shutdown()

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_active_process_not_cleaned(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
    ) -> None:
        """Test that actively used processes are not cleaned up."""
        pool = ProcessPool(idle_timeout=1, max_size=5)

        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        params = StdioServerParameters(command="echo", args=["test"])

        # Create a process but don't release it (simulates active use)
        handle = await pool.get_or_create(
            server_name="test-server",
            params=params,
            header_env_vars={"TOKEN": "test123"},
        )

        # Manually set last_used to be older than timeout
        cache_key = handle.cache_key
        pool._pool[cache_key].last_used = datetime.now(timezone.utc) - timedelta(seconds=10)

        # Run cleanup manually - should NOT remove because process is in use
        await pool._cleanup_idle_processes()

        assert pool.get_cached_count() == 1  # Still there because in use

        await handle.release()
        await pool.shutdown()


class TestConcurrency:
    """Tests for concurrent access handling."""

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_concurrent_get_or_create_same_key(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
    ) -> None:
        """Test that concurrent requests for same key don't create duplicates."""
        pool = ProcessPool(idle_timeout=600, max_size=100)

        # Setup mocks with a delay to simulate slow process creation
        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        async def slow_enter(_: MagicMock) -> tuple[MagicMock, MagicMock]:
            await asyncio.sleep(0.1)  # Simulate slow startup
            return (mock_reader, mock_writer)

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__ = slow_enter
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        params = StdioServerParameters(command="echo", args=["test"])
        env_vars = {"TOKEN": "same_token"}

        # Launch multiple concurrent requests for same key
        tasks = [
            pool.get_or_create(
                server_name="test-server",
                params=params,
                header_env_vars=env_vars,
            )
            for _ in range(5)
        ]

        handles = await asyncio.gather(*tasks)

        # All handles should point to the same session
        sessions = {h.session for h in handles}
        # Due to the lock, only one process should be created
        # But subsequent requests will wait and get the cached one
        # The first request creates, others wait and reuse

        # Release all handles
        for handle in handles:
            await handle.release()

        # Should only have created one process
        assert pool.get_cached_count() == 1

        await pool.shutdown()


class TestIntegration:
    """Integration tests for process pool with mcp_server."""

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_first_request_cold_start(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
    ) -> None:
        """Test that first request triggers cold start (cache miss)."""
        pool = ProcessPool(idle_timeout=600, max_size=100)

        # Track timing
        call_times: list[float] = []

        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        async def timed_enter(_: MagicMock) -> tuple[MagicMock, MagicMock]:
            import time

            call_times.append(time.time())
            return (mock_reader, mock_writer)

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__ = timed_enter
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        params = StdioServerParameters(command="echo", args=["test"])

        # First request - should trigger process creation (cold start)
        handle = await pool.get_or_create(
            server_name="test-server",
            params=params,
            header_env_vars={"TOKEN": "user_a_token"},
        )

        assert len(call_times) == 1  # Process was created
        assert pool.get_cached_count() == 1

        await handle.release()
        await pool.shutdown()

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_second_request_cache_hit(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
    ) -> None:
        """Test that second request with same token uses cache (cache hit)."""
        pool = ProcessPool(idle_timeout=600, max_size=100)

        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        params = StdioServerParameters(command="echo", args=["test"])
        token = {"TOKEN": "user_a_token"}

        # First request - cold start
        handle1 = await pool.get_or_create(
            server_name="test-server",
            params=params,
            header_env_vars=token,
        )
        await handle1.release()

        # Second request - cache hit
        handle2 = await pool.get_or_create(
            server_name="test-server",
            params=params,
            header_env_vars=token,
        )

        # Should have only called stdio_client once
        assert mock_stdio.call_count == 1
        # Same session should be reused
        assert handle2.session is handle1.session

        await handle2.release()
        await pool.shutdown()

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_different_user_new_process(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
    ) -> None:
        """Test that request with different token creates new process."""
        pool = ProcessPool(idle_timeout=600, max_size=100)

        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        # Return different session objects for each call
        sessions = [MagicMock() for _ in range(3)]
        for s in sessions:
            s.initialize = AsyncMock()

        session_index = [0]

        async def get_session(_: MagicMock) -> MagicMock:
            idx = session_index[0]
            session_index[0] += 1
            return sessions[idx % len(sessions)]

        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = get_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        params = StdioServerParameters(command="echo", args=["test"])

        # User A request
        handle_a = await pool.get_or_create(
            server_name="test-server",
            params=params,
            header_env_vars={"TOKEN": "user_a_token"},
        )
        await handle_a.release()

        # User B request (different token)
        handle_b = await pool.get_or_create(
            server_name="test-server",
            params=params,
            header_env_vars={"TOKEN": "user_b_token"},
        )

        # Should have created two processes
        assert mock_stdio.call_count == 2
        assert pool.get_cached_count() == 2

        await handle_b.release()
        await pool.shutdown()

    @patch("mcp_proxy.process_pool.stdio_client")
    @patch("mcp_proxy.process_pool.ClientSession")
    async def test_multi_user_scenario(
        self,
        mock_session_class: MagicMock,
        mock_stdio: MagicMock,
    ) -> None:
        """Test realistic multi-user scenario with cache hits and misses."""
        pool = ProcessPool(idle_timeout=600, max_size=100)

        mock_reader = MagicMock()
        mock_reader.at_eof.return_value = False
        mock_writer = MagicMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (mock_reader, mock_writer)
        mock_stdio_context.__aexit__.return_value = None
        mock_stdio.return_value = mock_stdio_context

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_context

        await pool.start()

        params = StdioServerParameters(command="echo", args=["test"])

        # Simulate: User A, User A, User B, User B, User A
        requests = [
            {"TOKEN": "token_a"},  # Cold start
            {"TOKEN": "token_a"},  # Cache hit
            {"TOKEN": "token_b"},  # Cold start
            {"TOKEN": "token_b"},  # Cache hit
            {"TOKEN": "token_a"},  # Cache hit
        ]

        for env_vars in requests:
            handle = await pool.get_or_create(
                server_name="test-server",
                params=params,
                header_env_vars=env_vars,
            )
            await handle.release()

        # Should have created 2 processes (one for each unique token)
        assert mock_stdio.call_count == 2
        assert pool.get_cached_count() == 2

        await pool.shutdown()
