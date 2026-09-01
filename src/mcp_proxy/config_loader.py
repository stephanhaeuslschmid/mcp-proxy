"""Configuration loader for MCP proxy.

This module provides functionality to load named server configurations from JSON files.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from mcp.client.stdio import StdioServerParameters

logger = logging.getLogger(__name__)


@dataclass
class ProxySettings:
    """Global settings for the MCP proxy."""

    process_pool_enabled: bool = True
    process_pool_idle_timeout: int = 600
    process_pool_max_size: int = 100


def load_settings_from_config(config_file_path: str | Path) -> ProxySettings:
    """Load global settings from config.json.

    Args:
        config_file_path: Path to the JSON configuration file.

    Returns:
        ProxySettings with values from config or defaults.
    """
    try:
        with Path(config_file_path).open() as f:
            config_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.debug("Could not load settings from config, using defaults")
        return ProxySettings()

    settings = config_data.get("settings", {})
    pool_settings = settings.get("processPool", {})

    return ProxySettings(
        process_pool_enabled=pool_settings.get("enabled", True),
        process_pool_idle_timeout=pool_settings.get("idleTimeout", 600),
        process_pool_max_size=pool_settings.get("maxSize", 100),
    )


def load_named_server_configs_from_file(
    config_file_path: str | Path,
    base_env: dict[str, str],
) -> tuple[dict[str, StdioServerParameters], dict[str, dict[str, str]], dict[str, list[str]]]:
    """Loads named server configurations from a JSON file.

    Args:
        config_file_path: Path to the JSON configuration file.
        base_env: The base environment dictionary to be inherited by servers.

    Returns:
        A tuple containing:
        - A dictionary of named server parameters
        - A dictionary of header-to-environment mappings for each server
        - A dictionary of header-to-args mappings for each server (list of header names)

    Raises:
        FileNotFoundError: If the config file is not found.
        json.JSONDecodeError: If the config file contains invalid JSON.
        ValueError: If the config file format is invalid.
    """
    named_stdio_params: dict[str, StdioServerParameters] = {}
    header_mappings: dict[str, dict[str, str]] = {}
    args_mappings: dict[str, list[str]] = {}
    logger.info("Loading named server configurations from: %s", config_file_path)

    try:
        with Path(config_file_path).open() as f:
            config_data = json.load(f)
    except FileNotFoundError:
        logger.exception("Configuration file not found: %s", config_file_path)
        raise
    except json.JSONDecodeError:
        logger.exception("Error decoding JSON from configuration file: %s", config_file_path)
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error opening or reading configuration file %s",
            config_file_path,
        )
        error_message = f"Could not read configuration file: {e}"
        raise ValueError(error_message) from e

    if not isinstance(config_data, dict) or "mcpServers" not in config_data:
        msg = f"Invalid config file format in {config_file_path}. Missing 'mcpServers' key."
        logger.error(msg)
        raise ValueError(msg)

    for name, server_config in config_data.get("mcpServers", {}).items():
        if not isinstance(server_config, dict):
            logger.warning(
                "Skipping invalid server config for '%s' in %s. Entry is not a dictionary.",
                name,
                config_file_path,
            )
            continue
        if not server_config.get("enabled", True):  # Default to True if 'enabled' is not present
            logger.info("Named server '%s' from config is not enabled. Skipping.", name)
            continue

        command = server_config.get("command")
        command_args = server_config.get("args", [])
        env = server_config.get("env", {})
        header_to_env = server_config.get("headerToEnv", {})
        header_to_args = server_config.get("headerToArgs", [])

        if not command:
            logger.warning(
                "Named server '%s' from config is missing 'command'. Skipping.",
                name,
            )
            continue
        if not isinstance(command_args, list):
            logger.warning(
                "Named server '%s' from config has invalid 'args' (must be a list). Skipping.",
                name,
            )
            continue
        if not isinstance(header_to_env, dict):
            logger.warning(
                "Named server '%s' from config has invalid 'headerToEnv' (must be a dict). Skipping.",
                name,
            )
            continue
        if not isinstance(header_to_args, list):
            logger.warning(
                "Named server '%s' from config has invalid 'headerToArgs' (must be a list). Skipping.",
                name,
            )
            continue

        new_env = base_env.copy()
        new_env.update(env)

        named_stdio_params[name] = StdioServerParameters(
            command=command,
            args=command_args,
            env=new_env,
            cwd=None,
        )

        # Store header mapping for this server
        if header_to_env:
            header_mappings[name] = header_to_env

        # Store header-to-args mapping for this server
        if header_to_args:
            args_mappings[name] = header_to_args

        logger.info(
            "Configured named server '%s' from config: %s %s (header env mappings: %s, header arg mappings: %s)",
            name,
            command,
            " ".join(command_args),
            list(header_to_env.keys()) if header_to_env else "none",
            header_to_args if header_to_args else "none",
        )

    return named_stdio_params, header_mappings, args_mappings
