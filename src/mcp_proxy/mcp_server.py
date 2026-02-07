"""Create a local SSE server that proxies requests to a stdio MCP server."""

import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import uvicorn
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server import Server as MCPServerSDK  # Renamed to avoid conflict
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import BaseRoute, Mount, Route
from starlette.types import Receive, Scope, Send

from .process_pool import (
    ProcessHandle,
    get_process_pool,
    init_process_pool,
    is_pool_enabled,
    shutdown_process_pool,
)
from .proxy_server import create_proxy_server

logger = logging.getLogger(__name__)


@dataclass
class MCPServerSettings:
    """Settings for the MCP server."""

    bind_host: str
    port: int
    stateless: bool = False
    allow_origins: list[str] | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


# To store last activity for multiple servers if needed, though status endpoint is global for now.
_global_status: dict[str, Any] = {
    "api_last_activity": datetime.now(timezone.utc).isoformat(),
    "server_instances": {},  # Could be used to store per-instance status later
}


def _update_global_activity() -> None:
    _global_status["api_last_activity"] = datetime.now(timezone.utc).isoformat()


class _ASGIEndpointAdapter:
    """Wrap a coroutine function into an ASGI application."""

    def __init__(self, endpoint: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self._endpoint = endpoint

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._endpoint(scope, receive, send)


HTTP_METHODS = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "TRACE"]


async def _handle_status(_: Request) -> Response:
    """Global health check and service usage monitoring endpoint."""
    return JSONResponse(_global_status)


def _extract_header_env_vars(
    request: Request,
    header_mapping: dict[str, str],
) -> dict[str, str]:
    """Extract environment variables from request headers based on mapping.

    Args:
        request: The incoming HTTP request
        header_mapping: Mapping of HTTP header names to environment variable names

    Returns:
        Dictionary of environment variable names to their values from headers
    """
    env_vars: dict[str, str] = {}
    for header_name, env_name in header_mapping.items():
        value = request.headers.get(header_name)
        if value:
            env_vars[env_name] = value
            logger.debug("Mapped header %s to env %s", header_name, env_name)
    return env_vars


def _extract_header_args(
    request: Request,
    header_to_args: list[str],
) -> list[str]:
    """Extract CLI arguments from request headers.

    Args:
        request: The incoming HTTP request
        header_to_args: List of HTTP header names to extract as CLI arguments

    Returns:
        List of argument values from headers (in order of header_to_args)
    """
    args: list[str] = []
    for header_name in header_to_args:
        value = request.headers.get(header_name)
        if value:
            args.append(value)
            logger.debug("Mapped header %s to arg", header_name)
    return args


def create_dynamic_server_routes(
    server_name: str,
    params: StdioServerParameters,
    header_mapping: dict[str, str],
    header_to_args: list[str],
    stateless_instance: bool,
) -> list[BaseRoute]:
    """Create routes for a server that spawns processes on-demand with header-based env vars.

    This is used for servers with headerToEnv or headerToArgs configuration.
    When is_pool_enabled(), processes with identical env vars and args are cached
    and reused. Otherwise, each request spawns a new stdio process with environment
    variables and/or arguments extracted from HTTP headers, executes the MCP operation,
    and then terminates the process.

    Args:
        server_name: Name of the server for logging
        params: Base stdio parameters (env vars and args from headers will be merged)
        header_mapping: Mapping of HTTP header names to environment variable names
        header_to_args: List of HTTP header names to extract as CLI arguments
        stateless_instance: Whether to run in stateless mode

    Returns:
        List of Starlette routes for this dynamic server
    """

    async def handle_dynamic_sse_pooled(request: Request) -> Response:
        """Handle SSE requests using pooled processes."""
        _update_global_activity()

        # Extract env vars and args from headers
        header_env_vars = _extract_header_env_vars(request, header_mapping)
        header_args = _extract_header_args(request, header_to_args)

        pool = get_process_pool()
        process_handle: ProcessHandle | None = None

        try:
            process_handle = await pool.get_or_create(
                server_name=server_name,
                params=params,
                header_env_vars=header_env_vars,
                extra_args=header_args,
            )

            proxy = await create_proxy_server(process_handle.session)

            sse_transport = SseServerTransport("/messages/")
            async with sse_transport.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
            ) as (read_stream, write_stream):
                await proxy.run(
                    read_stream,
                    write_stream,
                    proxy.create_initialization_options(),
                )
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(
                "Connection error for %s, invalidating cached process: %s",
                server_name,
                e,
            )
            if process_handle:
                await pool.invalidate(process_handle.cache_key)
            raise
        finally:
            if process_handle:
                await process_handle.release()

        return Response()

    async def handle_dynamic_mcp_pooled(scope: Scope, receive: Receive, send: Send) -> None:
        """Handle StreamableHTTP requests using pooled processes."""
        _update_global_activity()

        # Create a Request object to extract headers
        request = Request(scope, receive, send)
        header_env_vars = _extract_header_env_vars(request, header_mapping)
        header_args = _extract_header_args(request, header_to_args)

        pool = get_process_pool()
        process_handle: ProcessHandle | None = None

        try:
            process_handle = await pool.get_or_create(
                server_name=server_name,
                params=params,
                header_env_vars=header_env_vars,
                extra_args=header_args,
            )

            proxy = await create_proxy_server(process_handle.session)

            http_session_manager = StreamableHTTPSessionManager(
                app=proxy,
                event_store=None,
                json_response=True,
                stateless=stateless_instance,
            )

            async with http_session_manager.run():
                # Normalize path if needed
                updated_scope = scope
                if scope.get("type") == "http":
                    path = scope.get("path", "")
                    if path and path.rstrip("/").endswith("/mcp") and not path.endswith("/"):
                        updated_scope = dict(scope)
                        updated_scope["path"] = path + "/"
                        raw_path = scope.get("raw_path")
                        if raw_path:
                            if b"?" in raw_path:
                                path_part, query_part = raw_path.split(b"?", 1)
                                updated_scope["raw_path"] = (
                                    path_part.rstrip(b"/") + b"/?" + query_part
                                )
                            else:
                                updated_scope["raw_path"] = raw_path.rstrip(b"/") + b"/"

                await http_session_manager.handle_request(updated_scope, receive, send)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(
                "Connection error for %s, invalidating cached process: %s",
                server_name,
                e,
            )
            if process_handle:
                await pool.invalidate(process_handle.cache_key)
            raise
        finally:
            if process_handle:
                await process_handle.release()

    async def handle_dynamic_sse(request: Request) -> Response:
        """Handle SSE requests by spawning a process with header-derived env vars and args."""
        _update_global_activity()

        # Extract env vars and args from headers
        header_env_vars = _extract_header_env_vars(request, header_mapping)
        header_args = _extract_header_args(request, header_to_args)

        # Merge with base env vars
        merged_env = (params.env or {}).copy()
        merged_env.update(header_env_vars)

        # Merge with base args
        merged_args = list(params.args) + header_args

        dynamic_params = StdioServerParameters(
            command=params.command,
            args=merged_args,
            env=merged_env,
            cwd=params.cwd,
        )

        logger.info(
            "Spawning dynamic process for %s with %d header-derived env vars and %d extra args",
            server_name,
            len(header_env_vars),
            len(header_args),
        )

        # Spawn process, handle request, then cleanup
        async with contextlib.AsyncExitStack() as stack:
            stdio_streams = await stack.enter_async_context(stdio_client(dynamic_params))
            session = await stack.enter_async_context(ClientSession(*stdio_streams))
            proxy = await create_proxy_server(session)

            sse_transport = SseServerTransport("/messages/")
            async with sse_transport.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
            ) as (read_stream, write_stream):
                await proxy.run(
                    read_stream,
                    write_stream,
                    proxy.create_initialization_options(),
                )

        return Response()

    async def handle_dynamic_mcp(scope: Scope, receive: Receive, send: Send) -> None:
        """Handle StreamableHTTP requests by spawning a process with header-derived env vars and args."""
        _update_global_activity()

        # Create a Request object to extract headers
        request = Request(scope, receive, send)
        header_env_vars = _extract_header_env_vars(request, header_mapping)
        header_args = _extract_header_args(request, header_to_args)

        # Merge with base env vars
        merged_env = (params.env or {}).copy()
        merged_env.update(header_env_vars)

        # Merge with base args
        merged_args = list(params.args) + header_args

        dynamic_params = StdioServerParameters(
            command=params.command,
            args=merged_args,
            env=merged_env,
            cwd=params.cwd,
        )

        logger.info(
            "Spawning dynamic process for %s with %d header-derived env vars and %d extra args",
            server_name,
            len(header_env_vars),
            len(header_args),
        )

        async with contextlib.AsyncExitStack() as stack:
            stdio_streams = await stack.enter_async_context(stdio_client(dynamic_params))
            session = await stack.enter_async_context(ClientSession(*stdio_streams))
            proxy = await create_proxy_server(session)

            http_session_manager = StreamableHTTPSessionManager(
                app=proxy,
                event_store=None,
                json_response=True,
                stateless=stateless_instance,
            )

            async with http_session_manager.run():
                # Normalize path if needed
                updated_scope = scope
                if scope.get("type") == "http":
                    path = scope.get("path", "")
                    if path and path.rstrip("/").endswith("/mcp") and not path.endswith("/"):
                        updated_scope = dict(scope)
                        updated_scope["path"] = path + "/"
                        raw_path = scope.get("raw_path")
                        if raw_path:
                            if b"?" in raw_path:
                                path_part, query_part = raw_path.split(b"?", 1)
                                updated_scope["raw_path"] = (
                                    path_part.rstrip(b"/") + b"/?" + query_part
                                )
                            else:
                                updated_scope["raw_path"] = raw_path.rstrip(b"/") + b"/"

                await http_session_manager.handle_request(updated_scope, receive, send)

    # Choose handlers based on pool enabled status
    if is_pool_enabled():
        logger.info(
            "Process pool enabled for server %s",
            server_name,
        )
        mcp_handler = handle_dynamic_mcp_pooled
        sse_handler = handle_dynamic_sse_pooled
    else:
        mcp_handler = handle_dynamic_mcp
        sse_handler = handle_dynamic_sse

    routes = [
        Route(
            "/mcp",
            endpoint=_ASGIEndpointAdapter(mcp_handler),
            methods=HTTP_METHODS,
            include_in_schema=False,
        ),
        Mount("/mcp", app=mcp_handler),
        Route("/sse", endpoint=sse_handler),
    ]
    return routes


def create_single_instance_routes(
    mcp_server_instance: MCPServerSDK[object],
    *,
    stateless_instance: bool,
) -> tuple[list[BaseRoute], StreamableHTTPSessionManager]:  # Return the manager itself
    """Create Starlette routes and the HTTP session manager for a single MCP server instance."""
    logger.debug(
        "Creating routes for a single MCP server instance (stateless: %s)",
        stateless_instance,
    )

    sse_transport = SseServerTransport("/messages/")
    http_session_manager = StreamableHTTPSessionManager(
        app=mcp_server_instance,
        event_store=None,
        json_response=True,
        stateless=stateless_instance,
    )

    async def handle_sse_instance(request: Request) -> Response:
        async with sse_transport.connect_sse(
            request.scope,
            request.receive,
            request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            _update_global_activity()
            await mcp_server_instance.run(
                read_stream,
                write_stream,
                mcp_server_instance.create_initialization_options(),
            )
        return Response()

    async def handle_streamable_http_instance(scope: Scope, receive: Receive, send: Send) -> None:
        _update_global_activity()
        updated_scope = scope
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path and path.rstrip("/") == "/mcp" and not path.endswith("/"):
                updated_scope = dict(scope)
                normalized_path = path + "/"
                logger.debug(
                    "Normalized request path from '%s' to '%s' without redirect",
                    path,
                    normalized_path,
                )
                updated_scope["path"] = normalized_path

                raw_path = scope.get("raw_path")
                if raw_path:
                    if b"?" in raw_path:
                        path_part, query_part = raw_path.split(b"?", 1)
                        updated_scope["raw_path"] = path_part.rstrip(b"/") + b"/?" + query_part
                    else:
                        updated_scope["raw_path"] = raw_path.rstrip(b"/") + b"/"

        await http_session_manager.handle_request(updated_scope, receive, send)

    routes = [
        Route(
            "/mcp",
            endpoint=_ASGIEndpointAdapter(handle_streamable_http_instance),
            methods=HTTP_METHODS,
            include_in_schema=False,
        ),
        Mount("/mcp", app=handle_streamable_http_instance),
        Route("/sse", endpoint=handle_sse_instance),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ]
    return routes, http_session_manager


async def run_mcp_server(
    mcp_settings: MCPServerSettings,
    default_server_params: StdioServerParameters | None = None,
    named_server_params: dict[str, StdioServerParameters] | None = None,
    header_mappings: dict[str, dict[str, str]] | None = None,
    args_mappings: dict[str, list[str]] | None = None,
) -> None:
    """Run stdio client(s) and expose an MCP server with multiple possible backends.

    Args:
        mcp_settings: Server configuration settings
        default_server_params: Parameters for the default stdio server
        named_server_params: Parameters for named stdio servers
        header_mappings: Mapping of server names to header->env mappings.
            For servers with header mappings, HTTP headers will be extracted
            and passed as environment variables to the stdio process.
            Example: {"brave-search": {"X-Brave-Api-Key": "BRAVE_API_KEY"}}
        args_mappings: Mapping of server names to header->args mappings.
            For servers with args mappings, HTTP headers will be extracted
            and appended as CLI arguments to the stdio process.
            Example: {"filesystem": ["X-Workspace-Path"]}
    """
    if named_server_params is None:
        named_server_params = {}
    if header_mappings is None:
        header_mappings = {}
    if args_mappings is None:
        args_mappings = {}

    all_routes: list[BaseRoute] = [
        Route("/status", endpoint=_handle_status),  # Global status endpoint
    ]

    # Check if any server uses headerToEnv or headerToArgs (needs process pool)
    has_dynamic_servers = any(
        (name in header_mappings and header_mappings[name])
        or (name in args_mappings and args_mappings[name])
        for name in (named_server_params or {})
    )

    # Initialize process pool if enabled and needed
    if is_pool_enabled() and has_dynamic_servers:
        await init_process_pool()
        logger.info("Process pool initialized for dynamic servers")

    # Use AsyncExitStack to manage lifecycles of multiple components
    async with contextlib.AsyncExitStack() as stack:
        # Manage lifespans of all StreamableHTTPSessionManagers
        @contextlib.asynccontextmanager
        async def combined_lifespan(_app: Starlette) -> AsyncIterator[None]:
            logger.info("Main application lifespan starting...")
            # All http_session_managers' .run() are already entered into the stack
            yield
            logger.info("Main application lifespan shutting down...")
            # Shutdown process pool
            if is_pool_enabled() and has_dynamic_servers:
                await shutdown_process_pool()

        # Setup default server if configured
        if default_server_params:
            logger.info(
                "Setting up default server: %s %s",
                default_server_params.command,
                " ".join(default_server_params.args),
            )
            stdio_streams = await stack.enter_async_context(stdio_client(default_server_params))
            session = await stack.enter_async_context(ClientSession(*stdio_streams))
            proxy = await create_proxy_server(session)

            instance_routes, http_manager = create_single_instance_routes(
                proxy,
                stateless_instance=mcp_settings.stateless,
            )
            await stack.enter_async_context(http_manager.run())  # Manage lifespan by calling run()
            all_routes.extend(instance_routes)
            _global_status["server_instances"]["default"] = "configured"

        # Setup named servers
        for name, params in named_server_params.items():
            # Check if this server has header mappings or args mappings (dynamic mode)
            has_header_env = name in header_mappings and header_mappings[name]
            has_header_args = name in args_mappings and args_mappings[name]

            if has_header_env or has_header_args:
                logger.info(
                    "Setting up dynamic named server '%s' with headerToEnv=%s, headerToArgs=%s: %s %s",
                    name,
                    bool(has_header_env),
                    bool(has_header_args),
                    params.command,
                    " ".join(params.args),
                )
                # Create dynamic routes that spawn processes on-demand
                dynamic_routes = create_dynamic_server_routes(
                    server_name=name,
                    params=params,
                    header_mapping=header_mappings.get(name, {}),
                    header_to_args=args_mappings.get(name, []),
                    stateless_instance=mcp_settings.stateless,
                )
                server_mount = Mount(f"/servers/{name}", routes=dynamic_routes)
                all_routes.append(server_mount)
                _global_status["server_instances"][name] = "dynamic"
            else:
                # Static server - start process at app startup
                logger.info(
                    "Setting up static named server '%s': %s %s",
                    name,
                    params.command,
                    " ".join(params.args),
                )
                stdio_streams_named = await stack.enter_async_context(stdio_client(params))
                session_named = await stack.enter_async_context(ClientSession(*stdio_streams_named))
                proxy_named = await create_proxy_server(session_named)

                instance_routes_named, http_manager_named = create_single_instance_routes(
                    proxy_named,
                    stateless_instance=mcp_settings.stateless,
                )
                await stack.enter_async_context(
                    http_manager_named.run(),
                )  # Manage lifespan by calling run()

                # Mount these routes under /servers/<name>/
                server_mount = Mount(f"/servers/{name}", routes=instance_routes_named)
                all_routes.append(server_mount)
                _global_status["server_instances"][name] = "static"

        if not default_server_params and not named_server_params:
            logger.error("No servers configured to run.")
            return

        middleware: list[Middleware] = []
        if mcp_settings.allow_origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=mcp_settings.allow_origins,
                    allow_methods=["*"],
                    allow_headers=["*"],
                ),
            )

        starlette_app = Starlette(
            debug=(mcp_settings.log_level == "DEBUG"),
            routes=all_routes,
            middleware=middleware,
            lifespan=combined_lifespan,
        )

        starlette_app.router.redirect_slashes = False

        config = uvicorn.Config(
            starlette_app,
            host=mcp_settings.bind_host,
            port=mcp_settings.port,
            log_level=mcp_settings.log_level.lower(),
        )
        http_server = uvicorn.Server(config)

        # Print out the SSE URLs for all configured servers
        base_url = f"http://{mcp_settings.bind_host}:{mcp_settings.port}"
        sse_urls = []

        # Add default server if configured
        if default_server_params:
            sse_urls.append(f"{base_url}/sse")

        # Add named servers
        sse_urls.extend([f"{base_url}/servers/{name}/sse" for name in named_server_params])

        # Display the SSE URLs prominently
        if sse_urls:
            # Using print directly for user visibility, with noqa to ignore linter warnings
            logger.info("Serving MCP Servers via SSE:")
            for url in sse_urls:
                logger.info("  - %s", url)

        logger.debug(
            "Serving incoming MCP requests on %s:%s",
            mcp_settings.bind_host,
            mcp_settings.port,
        )
        await http_server.serve()
