"""Configuration: pydantic models mirroring ``config.toml`` field-for-field.

Compatibility contract: field names and nesting must match the Rust project's
``src/config.rs`` exactly, so an existing ``config.toml`` loads unchanged.

Several comments in the Rust repo's ``config.example.toml`` are stale. The
defaults here follow ``src/config.rs``, which is what actually runs:

    session_policy          "reuse"   (example file says per_request)
    startup_health_check    False     (example file says true)
    default_search_enabled  False     (example file says true)
"""

from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_API_BASE = "https://chat.deepseek.com/api/v0"
DEFAULT_WASM_URL = (
    "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"
)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    host: str = "127.0.0.1"
    port: int = 22217
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:22217"])


class AccountConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    email: str = ""
    mobile: str = ""
    area_code: str = ""
    password: str = ""
    device_id: str = ""

    @property
    def identifier(self) -> str:
        return self.email.strip() or self.mobile.strip()


class ToolCallConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    extra_starts: list[str] = Field(
        default_factory=lambda: ["<|tool_call_begin|>", "<tool_calls>", "<tool_call>"]
    )
    extra_ends: list[str] = Field(
        default_factory=lambda: ["<|tool_call_end|>", "</tool_calls>", "</tool_call>"]
    )


class DsCoreConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    accounts: list[AccountConfig] = Field(default_factory=list)

    api_base: str = DEFAULT_API_BASE
    wasm_url: str = DEFAULT_WASM_URL

    user_agent: str = DEFAULT_USER_AGENT
    client_version: str = "2.5.0"
    client_platform: str = "web"
    client_locale: str = "zh_CN"
    client_bundle_id: str = "com.deepseek.chat"
    client_device_id: str = ""
    client_device_model: str = ""
    client_timezone_offset: str = "28800"
    client_os: str = "web"

    emulation: str = "chrome136"
    hif_enabled: bool = True
    startup_health_check: bool = False

    session_policy: str = "reuse"
    session_idle_secs: int = 900

    model_types: list[str] = Field(default_factory=lambda: ["default"])
    max_input_tokens: list[int] = Field(default_factory=lambda: [1048576])
    max_output_tokens: list[int] = Field(default_factory=lambda: [384000])
    input_character_limits: list[int] = Field(default_factory=lambda: [2621440])
    model_aliases: list[str] = Field(default_factory=list)

    tool_call: ToolCallConfig = Field(default_factory=ToolCallConfig)
    #: Extension over the Rust project: which tool-call dialect to emit/parse.
    #: ``dsml`` (default) targets DeepSeek V4+, ``legacy`` reproduces the old
    #: ``<|tool▁calls▁begin|>`` protocol.
    tool_protocol: str = "dsml"

    hourly_request_quota: int = 60
    default_search_enabled: bool = False

    responses_store_capacity: int = 256
    responses_store_ttl_secs: int = 3600


class ProxyConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str | None = None


class AdminConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    password_hash: str = ""
    jwt_secret: str = ""
    jwt_issued_at: int = 0


class ApiKeyEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    description: str = ""


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    server: ServerConfig = Field(default_factory=ServerConfig)
    ds_core: DsCoreConfig = Field(default_factory=DsCoreConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    api_keys: list[ApiKeyEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> Config:
        self.validate_config()
        return self

    # -- validation --------------------------------------------------------

    def validate_config(self) -> None:
        core = self.ds_core
        if not core.model_types:
            raise ValueError("ds_core.model_types must not be empty")
        expected = len(core.model_types)
        for name in ("max_input_tokens", "max_output_tokens", "input_character_limits"):
            values = getattr(core, name)
            if len(values) != expected:
                raise ValueError(
                    f"ds_core.{name} must have {expected} entries to align with model_types"
                )
        if core.model_aliases and len(core.model_aliases) != expected:
            raise ValueError(
                f"ds_core.model_aliases must have {expected} entries or be empty"
            )
        seen: set[str] = set()
        for entry in self.api_keys:
            if entry.key in seen:
                raise ValueError(f"duplicate api key: {entry.key[:8]}...")
            seen.add(entry.key)

    # -- model registry ----------------------------------------------------

    @property
    def model_ids(self) -> list[str]:
        """Everything we advertise on /v1/models, in a stable order."""
        core = self.ds_core
        ids = [f"deepseek-{ty}" for ty in core.model_types]
        ids.extend(alias for alias in core.model_aliases if alias)
        ids.extend(core.model_types)
        out: list[str] = []
        seen: set[str] = set()
        for model_id in ids:
            if model_id.lower() in seen:
                continue
            seen.add(model_id.lower())
            out.append(model_id)
        return out

    @property
    def model_registry(self) -> dict[str, str]:
        """Lower-cased model id -> upstream ``model_type``."""
        core = self.ds_core
        registry: dict[str, str] = {}
        for index, model_type in enumerate(core.model_types):
            registry[f"deepseek-{model_type}".lower()] = model_type
            if index < len(core.model_aliases):
                alias = core.model_aliases[index]
                if alias:
                    registry[alias.lower()] = model_type
            registry[model_type.lower()] = model_type
        return registry

    def resolve_model_type(self, model: str | None) -> str:
        if not model:
            return self.ds_core.model_types[0]
        return self.model_registry.get(model.lower(), self.ds_core.model_types[0])

    def limits_for(self, model_type: str) -> tuple[int, int, int]:
        """Return ``(max_input_tokens, max_output_tokens, input_character_limit)``."""
        core = self.ds_core
        try:
            index = core.model_types.index(model_type)
        except ValueError:
            index = 0
        return (
            core.max_input_tokens[index],
            core.max_output_tokens[index],
            core.input_character_limits[index],
        )

    def is_valid_api_key(self, key: str) -> bool:
        return any(entry.key == key for entry in self.api_keys)

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> Config:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    @classmethod
    def load_with_args(cls, argv: list[str] | None = None) -> tuple[Config, Path]:
        """Resolve the config path from ``-c`` / ``DS_CONFIG_PATH`` and load it."""
        argv = list(argv if argv is not None else [])
        path: Path | None = None
        for index, arg in enumerate(argv):
            if arg in ("-c", "--config") and index + 1 < len(argv):
                path = Path(argv[index + 1])
                break
        if path is None:
            env = os.environ.get("DS_CONFIG_PATH")
            if env:
                path = Path(env)
        if path is None:
            path = Path("config.toml")

        if not path.exists():
            explicit = any(a in ("-c", "--config") for a in argv) or os.environ.get(
                "DS_CONFIG_PATH"
            )
            if explicit:
                raise FileNotFoundError(f"config file not found: {path}")
            config = cls()
            config.save(path)
            return config, path
        return cls.load(path), path

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
        data: dict[str, Any] = self.model_dump(mode="python")
        if data.get("proxy", {}).get("url") is None:
            data["proxy"] = {}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(tomli_w.dumps(data), encoding="utf-8")
        os.replace(tmp, path)


def data_dir() -> Path:
    return Path(os.environ.get("DS_DATA_DIR", ".")).resolve()
