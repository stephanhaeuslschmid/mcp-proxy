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

# Configuration from environment variables (can be overridden by config.json)
_DEFAULT_IDLE_TIMEOUT = int(os.environ.get("PROCESS_POOL_IDLE_TIMEOUT", "600"))
_DEFAULT_MAX_SIZE = int(os.environ.get("PROCESS_POOL_MAX_SIZE", "100"))
_DEFAULT_ENABLED = os.environ.get("PROCESS_POOL_ENABLED", "true").lower() == "true"
PROCESS_POOL_CLEANUP_INTERVAL = 60  # Check for idle processes every 60 seconds

# Runtime configuration (can be updated via configure_pool)
_pool_enabled: bool = _DEFAULT_ENABLED
_pool_idle_timeout: int = _DEFAULT_IDLE_TIMEOUT
_pool_max_size: int = _DEFAULT_MAX_SIZE

# Legacy exports for backwards compatibility
PROCESS_POOL_IDLE_TIMEOUT = _DEFAULT_IDLE_TIMEOUT
PROCESS_POOL_MAX_SIZE = _DEFAULT_MAX_SIZE
PROCESS_POOL_ENABLED = _DEFAULT_ENABLED


def configure_pool(
    enabled: bool | None = None,
    idle_timeout: int | None = None,
    max_size: int | None = None,
) -> None:
    """Configure pool settings from config.json (overrides ENV defaults).

    Args:
        enabled: Whether the process pool is enabled
        idle_timeout: Seconds after which idle processes are terminated
        max_size: Maximum number of processes to cache
    """
    global _pool_enabled, _pool_idle_timeout, _pool_max_size  # noqa: PLW0603

    if enabled is not None:
        _pool_enabled = enabled
        logger.info("Process pool enabled: %s (from config)", enabled)
    if idle_timeout is not None:
        _pool_idle_timeout = idle_timeout
        logger.info("Process pool idle timeout: %ds (from config)", idle_timeout)
    if max_size is not None:
        _pool_max_size = max_size
        logger.info("Process pool max size: %d (from config)", max_size)


def is_pool_enabled() -> bool:
    """Check if the process pool is enabled.

    Returns:
        True if process pooling is enabled, False otherwise.
    """
    return _pool_enabled


def get_pool_idle_timeout() -> int:
    """Get the configured idle timeout for pooled processes.

    Returns:
        Idle timeout in seconds.
    """
    return _pool_idle_timeout


def get_pool_max_size() -> int:
    """Get the configured maximum pool size.

    Returns:
        Maximum number of processes to cache.
    """
    return _pool_max_size


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
    _marked_for_removal: bool = field(default=False, init=False)

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

    def mark_for_removal(self) -> None:
        """Mark this process for lazy cleanup in the next request-task context."""
        self._marked_for_removal = True

    @property
    def is_marked_for_removal(self) -> bool:
        """Check if process is marked for removal."""
        return self._marked_for_removal


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


def generate_cache_key(
    server_name: str,
    env_vars: dict[str, str],
    extra_args: list[str] | None = None,
) -> str:
    """Generate deterministic cache key from server name, env vars, and extra args.

    Args:
        server_name: Name of the MCP server
        env_vars: Environment variables for the process
        extra_args: Additional CLI arguments for the process

    Returns:
        SHA256 hash of the server name, sorted env vars, and extra args
    """
    # Sort env vars for consistent hashing
    sorted_env = json.dumps(dict(sorted(env_vars.items())), sort_keys=True)
    args_str = json.dumps(extra_args or [])
    key_string = f"{server_name}:{sorted_env}:{args_str}"
    return hashlib.sha256(key_string.encode()).hexdigest()


class ProcessPool:
    """Pool for reusing MCP server processes with identical env vars.

    This pool caches stdio processes based on their environment variables,
    allowing reuse of processes when the same user makes multiple requests.
    """

    def __init__(
        self,
        idle_timeout: int | None = None,
        max_size: int | None = None,
    ) -> None:
        """Initialize the process pool.

        Args:
            idle_timeout: Seconds after which idle processes are terminated (uses global config if None)
            max_size: Maximum number of processes to cache (uses global config if None)
        """
        self._pool: dict[str, CachedProcess] = {}
        self._lock = asyncio.Lock()
        self._idle_timeout = idle_timeout if idle_timeout is not None else get_pool_idle_timeout()
        self._max_size = max_size if max_size is not None else get_pool_max_size()
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
        """Mark idle processes for lazy cleanup.

        Note: We only mark processes here, not actually close them.
        The actual cleanup happens in get_or_create() which runs in the
        request-task context. This avoids anyio's cancel scope issues
        that occur when closing AsyncExitStack from a different task.
        See: https://github.com/modelcontextprotocol/python-sdk/issues/577
        """
        now = datetime.now(timezone.utc)
        marked_count = 0

        async with self._lock:
            for cached in self._pool.values():
                if cached.is_marked_for_removal:
                    continue  # Already marked
                idle_seconds = (now - cached.last_used).total_seconds()
                if idle_seconds > self._idle_timeout and not cached.is_in_use:
                    cached.mark_for_removal()
                    marked_count += 1
                    logger.info(
                        "Process for %s idle for %.0fs, marked for lazy cleanup",
                        cached.server_name,
                        idle_seconds,
                    )

        if marked_count:
            logger.debug(
                "Process pool: marked %d idle processes for cleanup",
                marked_count,
            )

    async def _cleanup_marked_processes(self) -> None:
        """Clean up processes marked for removal.

        This runs in the request-task context, which is the same context
        where the processes were created. This avoids anyio's cancel scope
        issues that occur when closing AsyncExitStack from a different task.
        """
        # Collect marked processes under lock, then close outside lock
        to_cleanup: list[CachedProcess] = []

        async with self._lock:
            marked_keys = [
                key
                for key, cached in self._pool.items()
                if cached.is_marked_for_removal and not cached.is_in_use
            ]
            for key in marked_keys:
                to_cleanup.append(self._pool.pop(key))

        # Close outside lock to avoid blocking other requests
        for cached in to_cleanup:
            try:
                await cached.exit_stack.aclose()
                logger.info(
                    "Lazy cleanup completed for %s (key: %s)",
                    cached.server_name,
                    cached.cache_key[:8],
                )
            except Exception:
                logger.exception(
                    "Error during lazy cleanup for %s",
                    cached.server_name,
                )

    async def get_or_create(
        self,
        server_name: str,
        params: StdioServerParameters,
        header_env_vars: dict[str, str],
        extra_args: list[str] | None = None,
    ) -> ProcessHandle:
        """Get a cached process or create a new one.

        Args:
            server_name: Name of the MCP server
            params: Base stdio parameters (command, args, base env)
            header_env_vars: Environment variables extracted from headers
            extra_args: Additional CLI arguments extracted from headers

        Returns:
            ProcessHandle for communicating with the MCP server
        """
        if extra_args is None:
            extra_args = []

        # First, clean up any processes marked for removal (in request-task context)
        await self._cleanup_marked_processes()

        # Merge base env with header env vars for cache key
        merged_env = (params.env or {}).copy()
        merged_env.update(header_env_vars)
        cache_key = generate_cache_key(server_name, merged_env, extra_args)

        # Check for marked process that needs cleanup (outside lock for aclose)
        marked_process_to_cleanup: CachedProcess | None = None

        async with self._lock:
            # Check if we have a cached process
            if cache_key in self._pool:
                cached = self._pool[cache_key]

                # Skip if marked for removal - clean it up and create new
                if cached.is_marked_for_removal:
                    if not cached.is_in_use:
                        # Safe to remove and cleanup
                        marked_process_to_cleanup = self._pool.pop(cache_key)
                        logger.info(
                            "Cached process for %s is marked for removal, will cleanup and create new",
                            server_name,
                        )
                    else:
                        # Still in use by another request, leave it for later cleanup
                        logger.info(
                            "Cached process for %s is marked but still in use, creating new",
                            server_name,
                        )
                else:
                    # Process is healthy, use it
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

        # Cleanup marked process outside lock (in request-task context)
        if marked_process_to_cleanup:
            try:
                await marked_process_to_cleanup.exit_stack.aclose()
                logger.info(
                    "Cleaned up marked process for %s (key: %s)",
                    marked_process_to_cleanup.server_name,
                    marked_process_to_cleanup.cache_key[:8],
                )
            except Exception:
                logger.exception(
                    "Error cleaning up marked process for %s",
                    marked_process_to_cleanup.server_name,
                )

        # Cache miss - create new process
        logger.info(
            "Process cache miss for %s, starting new process",
            server_name,
        )

        # Create new process with merged env and extra args
        merged_args = list(params.args) + extra_args
        dynamic_params = StdioServerParameters(
            command=params.command,
            args=merged_args,
            env=merged_env,
            cwd=params.cwd,
        )

        exit_stack = contextlib.AsyncExitStack()
        try:
            stdio_streams = await exit_stack.enter_async_context(stdio_client(dynamic_params))
            session = await exit_stack.enter_async_context(ClientSession(*stdio_streams))

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

            # Add to cache if not at max size (need lock for pool modification)
            async with self._lock:
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
        # Note: This may trigger anyio cancel scope errors since we're in a
        # different task than where the processes were created. However, the
        # subprocesses will still be terminated by the OS when we exit.
        # See: https://github.com/modelcontextprotocol/python-sdk/issues/577
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
                except RuntimeError as e:
                    if "cancel scope" in str(e):
                        # Known anyio limitation - process will still be cleaned up by OS
                        logger.debug(
                            "Ignoring anyio cancel scope error during shutdown for %s (known issue)",
                            cached.server_name,
                        )
                    else:
                        logger.exception(
                            "Error shutting down process for %s",
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
