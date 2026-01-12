"""Process pool for caching and reusing MCP server processes with identical env vars."""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

# Configuration from environment variables
PROCESS_POOL_IDLE_TIMEOUT = int(os.environ.get("PROCESS_POOL_IDLE_TIMEOUT", "600"))
PROCESS_POOL_MAX_SIZE = int(os.environ.get("PROCESS_POOL_MAX_SIZE", "100"))
PROCESS_POOL_ENABLED = os.environ.get("PROCESS_POOL_ENABLED", "true").lower() == "true"
PROCESS_POOL_CLEANUP_INTERVAL = 60  # Check for idle processes every 60 seconds


@dataclass
class CachedProcess:
    """Represents a cached MCP server process with its session."""

    server_name: str
    cache_key: str
    created_at: datetime
    last_used: datetime
    exit_stack: contextlib.AsyncExitStack
    session: ClientSession
    _reference_count: int = field(default=0, init=False)

    def increment_ref(self) -> None:
        """Increment reference count when process is being used."""
        self._reference_count += 1

    def decrement_ref(self) -> None:
        """Decrement reference count when process usage is complete."""
        self._reference_count -= 1

    @property
    def is_in_use(self) -> bool:
        """Check if process is currently being used."""
        return self._reference_count > 0


@dataclass
class ProcessHandle:
    """Handle returned to callers for using a cached process."""

    session: ClientSession
    cache_key: str
    _pool: "ProcessPool"
    _cached_process: CachedProcess

    async def release(self) -> None:
        """Release this handle, decrementing the reference count."""
        self._cached_process.decrement_ref()

    async def __aenter__(self) -> "ProcessHandle":
        """Enter async context."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit async context and release the handle."""
        await self.release()


def generate_cache_key(server_name: str, env_vars: dict[str, str]) -> str:
    """Generate deterministic cache key from server name and env vars.

    Args:
        server_name: Name of the MCP server
        env_vars: Environment variables for the process

    Returns:
        SHA256 hash of the server name and sorted env vars
    """
    # Sort env vars for consistent hashing
    sorted_env = json.dumps(dict(sorted(env_vars.items())), sort_keys=True)
    key_string = f"{server_name}:{sorted_env}"
    return hashlib.sha256(key_string.encode()).hexdigest()


class ProcessPool:
    """Pool for reusing MCP server processes with identical env vars.

    This pool caches stdio processes based on their environment variables,
    allowing reuse of processes when the same user makes multiple requests.
    """

    def __init__(
        self,
        idle_timeout: int = PROCESS_POOL_IDLE_TIMEOUT,
        max_size: int = PROCESS_POOL_MAX_SIZE,
    ) -> None:
        """Initialize the process pool.

        Args:
            idle_timeout: Seconds after which idle processes are terminated
            max_size: Maximum number of processes to cache
        """
        self._pool: dict[str, CachedProcess] = {}
        self._lock = asyncio.Lock()
        self._idle_timeout = idle_timeout
        self._max_size = max_size
        self._cleanup_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._started = False

    async def start(self) -> None:
        """Start the background cleanup task."""
        if self._started:
            return
        self._started = True
        self._shutdown_event.clear()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(
            "Process pool started (idle_timeout=%ds, max_size=%d)",
            self._idle_timeout,
            self._max_size,
        )

    async def _cleanup_loop(self) -> None:
        """Background task that periodically cleans up idle processes."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for cleanup interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=PROCESS_POOL_CLEANUP_INTERVAL,
                    )
                    # If we get here, shutdown was requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, proceed with cleanup
                    pass

                await self._cleanup_idle_processes()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in cleanup loop")

    async def _cleanup_idle_processes(self) -> None:
        """Remove and terminate processes that have been idle too long."""
        now = datetime.now(timezone.utc)
        to_remove: list[str] = []

        async with self._lock:
            for cache_key, cached in self._pool.items():
                idle_seconds = (now - cached.last_used).total_seconds()
                if idle_seconds > self._idle_timeout and not cached.is_in_use:
                    to_remove.append(cache_key)
                    logger.info(
                        "Process for %s idle for %.0fs, marking for removal",
                        cached.server_name,
                        idle_seconds,
                    )

            for cache_key in to_remove:
                cached = self._pool.pop(cache_key)
                try:
                    await cached.exit_stack.aclose()
                    logger.info(
                        "Cleaned up idle process for %s (key: %s)",
                        cached.server_name,
                        cache_key[:8],
                    )
                except Exception:
                    logger.exception(
                        "Error closing idle process for %s",
                        cached.server_name,
                    )

        if to_remove:
            logger.debug(
                "Process pool cleanup: removed %d idle processes, %d remaining",
                len(to_remove),
                len(self._pool),
            )

    async def get_or_create(
        self,
        server_name: str,
        params: StdioServerParameters,
        header_env_vars: dict[str, str],
    ) -> ProcessHandle:
        """Get a cached process or create a new one.

        Args:
            server_name: Name of the MCP server
            params: Base stdio parameters (command, args, base env)
            header_env_vars: Environment variables extracted from headers

        Returns:
            ProcessHandle for communicating with the MCP server
        """
        # Merge base env with header env vars for cache key
        merged_env = (params.env or {}).copy()
        merged_env.update(header_env_vars)
        cache_key = generate_cache_key(server_name, merged_env)

        async with self._lock:
            # Check if we have a cached process
            if cache_key in self._pool:
                cached = self._pool[cache_key]

                # Assume process is alive - errors will be caught by caller
                # and trigger invalidation via the error handling in mcp_server.py
                cached.last_used = datetime.now(timezone.utc)
                cached.increment_ref()
                logger.info(
                    "Process cache hit for %s (key: %s)",
                    server_name,
                    cache_key[:8],
                )
                return ProcessHandle(
                    session=cached.session,
                    cache_key=cache_key,
                    _pool=self,
                    _cached_process=cached,
                )

            # Cache miss - create new process
            logger.info(
                "Process cache miss for %s, starting new process",
                server_name,
            )

            # Create new process with merged env
            dynamic_params = StdioServerParameters(
                command=params.command,
                args=params.args,
                env=merged_env,
                cwd=params.cwd,
            )

            exit_stack = contextlib.AsyncExitStack()
            try:
                stdio_streams = await exit_stack.enter_async_context(
                    stdio_client(dynamic_params)
                )
                session = await exit_stack.enter_async_context(
                    ClientSession(*stdio_streams)
                )

                # Initialize the session
                await session.initialize()

                now = datetime.now(timezone.utc)
                cached_process = CachedProcess(
                    server_name=server_name,
                    cache_key=cache_key,
                    created_at=now,
                    last_used=now,
                    exit_stack=exit_stack,
                    session=session,
                )
                cached_process.increment_ref()

                # Add to cache if not at max size
                if len(self._pool) < self._max_size:
                    self._pool[cache_key] = cached_process
                    logger.debug(
                        "Process pool: %d cached processes",
                        len(self._pool),
                    )
                else:
                    logger.warning(
                        "Process pool at max size (%d), not caching new process for %s",
                        self._max_size,
                        server_name,
                    )

                return ProcessHandle(
                    session=session,
                    cache_key=cache_key,
                    _pool=self,
                    _cached_process=cached_process,
                )
            except Exception:
                await exit_stack.aclose()
                raise

    async def invalidate(self, cache_key: str) -> None:
        """Remove and terminate a cached process.

        Args:
            cache_key: The cache key of the process to invalidate
        """
        async with self._lock:
            if cache_key in self._pool:
                cached = self._pool.pop(cache_key)
                logger.info(
                    "Invalidating process for %s (key: %s)",
                    cached.server_name,
                    cache_key[:8],
                )
                try:
                    await cached.exit_stack.aclose()
                except Exception:
                    logger.exception(
                        "Error closing invalidated process for %s",
                        cached.server_name,
                    )

    def get_cached_count(self) -> int:
        """Get the number of currently cached processes.

        Returns:
            Number of processes in the pool
        """
        return len(self._pool)

    async def shutdown(self) -> None:
        """Gracefully shutdown all cached processes."""
        logger.info("Process pool shutdown initiated")

        # Signal cleanup loop to stop
        self._shutdown_event.set()

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all cached processes
        async with self._lock:
            for cache_key, cached in list(self._pool.items()):
                logger.debug(
                    "Shutting down process for %s (key: %s)",
                    cached.server_name,
                    cache_key[:8],
                )
                try:
                    await asyncio.wait_for(cached.exit_stack.aclose(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for process %s to terminate",
                        cached.server_name,
                    )
                except Exception:
                    logger.exception(
                        "Error shutting down process for %s",
                        cached.server_name,
                    )

            self._pool.clear()

        self._started = False
        logger.info("Process pool shutdown complete")


# Global process pool instance
_process_pool: ProcessPool | None = None


def get_process_pool() -> ProcessPool:
    """Get or create the global process pool instance.

    Returns:
        The global ProcessPool instance
    """
    global _process_pool  # noqa: PLW0603
    if _process_pool is None:
        _process_pool = ProcessPool()
    return _process_pool


async def init_process_pool() -> ProcessPool:
    """Initialize and start the global process pool.

    Returns:
        The started ProcessPool instance
    """
    pool = get_process_pool()
    await pool.start()
    return pool


async def shutdown_process_pool() -> None:
    """Shutdown the global process pool if it exists."""
    global _process_pool  # noqa: PLW0603
    if _process_pool is not None:
        await _process_pool.shutdown()
        _process_pool = None
