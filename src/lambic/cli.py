"""CLI entry point for lambic."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lambic",
        description="MCP-aware LLM shell with provider switching",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="",
        help="Path to config TOML file",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="",
        help="LLM model spec (e.g. ollama/qwen3.5:9b)",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable reasoning/thinking mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Max output tokens (default: 16384)",
    )
    parser.add_argument(
        "--log",
        default="",
        help="Log file path",
    )
    args = parser.parse_args()

    from lambic.core.config import ShellConfig

    # Load config
    config_path = args.config or ShellConfig.default_path()
    config = ShellConfig.from_toml(config_path)

    # CLI overrides
    if args.model:
        if "/" in args.model:
            provider, model = args.model.split("/", 1)
        else:
            provider, model = "ollama", args.model
        config.llm.provider = provider
        config.llm.model = model

    if args.no_think:
        config.llm.think = False

    if args.max_tokens:
        config.llm.max_tokens = args.max_tokens

    if args.log:
        config.log_file = args.log

    # Set up logging
    import logging

    handlers: list[logging.Handler] = []
    if config.log_file:
        handlers.append(logging.FileHandler(config.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers or None,
    )

    from lambic.tui.app import Shell

    shell = Shell(config=config)
    shell.run()


if __name__ == "__main__":
    main()
