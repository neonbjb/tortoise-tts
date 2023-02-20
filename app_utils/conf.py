import shelve
from typing import Any

from pydantic import BaseModel


class PersistentSettings(BaseModel):
    """
    This pydantic model will try to initialize itself from
    the database upon every instantiation

    It further supplies an update function, that allows to write
    back any changes into the database, under its key.
    """

    def __init__(self, **data: Any):
        with shelve.open("config.db") as db:
            super().__init__(**db.get("settings", default={}), **data)

    def update(self, **data: Any) -> None:
        """
        Persist the pydantic-dict that represents the model
        """
        with shelve.open("config.db") as db:
            db["settings"] = {**self.dict(), **data}


class TortoiseConfig(PersistentSettings):
    EXTRA_VOICES_DIR: str = ""
    AR_CHECKPOINT: str = "."
    DIFF_CHECKPOINT: str = "."
    LOW_VRAM: bool = True
