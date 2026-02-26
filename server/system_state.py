import json
import os
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
STATE_FILE = os.path.join(BASE_DIR, "sistema.json")

class SystemState:
    def __init__(self):
        self.data: Dict[str, Any] = {
            "last_update": "",
            "frontend": {},
            "market": {},
            "analysis": {}
        }
        self.load()

    def reset(self):
        self.data = {
            "last_update": "",
            "frontend": {},
            "market": {},
            "analysis": {},
            "pattern_notes": {}
        }
        self.save()

    def load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    self.data = json.load(f)
            except:
                pass

    def save(self):
        try:
            self.data["last_update"] = datetime.now().isoformat()
            tmp_file = STATE_FILE + ".tmp"
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
            try:
                os.replace(tmp_file, STATE_FILE)
            except Exception as re:
                logger.error(f"Atomic replace failed: {re}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def update(self, key: str, value: Any) -> None:
        self.data[key] = value

system_state = SystemState()

async def state_persister():
    while True:
        system_state.save()
        await asyncio.sleep(5)