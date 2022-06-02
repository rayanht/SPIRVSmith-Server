from typing import Literal

from pydantic import BaseSettings


class Settings(BaseSettings):
    app_env: Literal["dev", "prod"] = "dev"
