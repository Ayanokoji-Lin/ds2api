"""Entry point: ``python -m ds2api [-c config.toml]``."""

from __future__ import annotations

import logging
import os
import sys

from .settings import Config, data_dir


def setup_logging() -> None:
    level_name = os.environ.get("DS_LOG", "info").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s %(levelname)-5s  %(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    # curl_cffi's asyncio backend uses add_reader, unsupported by Proactor.
    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        config, path = Config.load_with_args(argv)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("failed to load config: %s", exc)
        return 2

    logger = logging.getLogger("ds2api")
    logger.info("config loaded from %s", path)
    if not config.ds_core.accounts:
        logger.warning("no accounts configured; every /v1 request will return 429")
    if not config.api_keys:
        logger.warning("no api keys configured; every /v1 request will return 401")

    import uvicorn

    from .server.app import create_app

    app = create_app(config, path)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=os.environ.get("DS_LOG", "info").lower(),
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
