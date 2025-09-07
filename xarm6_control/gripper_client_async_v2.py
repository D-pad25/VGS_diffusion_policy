#!/usr/bin/env python3
"""
Async JSON line protocol client for normalized-position gripper control.
"""

import asyncio, threading, concurrent.futures
import json
from typing import Optional


class GripperClientAsync:
    def __init__(self, host: str = "127.0.0.1", port: int = 22345, read_timeout: float = 5.0):
        self.host = host
        self.port = port
        self.read_timeout = read_timeout
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self):
        if self._reader is None or self._writer is None:
            self._reader, self._writer = await asyncio.open_connection(self.host, self.port)

    async def disconnect(self):
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            finally:
                self._reader = self._writer = None

    async def _rpc(self, payload: dict) -> dict:
        await self.connect()
        self._writer.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
        await self._writer.drain()
        try:
            data = await asyncio.wait_for(self._reader.readline(), timeout=self.read_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError("read timeout")
        if not data:
            raise ConnectionError("connection closed by server")
        return json.loads(data.decode())

    async def set(self, value: float) -> dict:
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError("value must be within [0,1]")
        return await self._rpc({"cmd": "SET", "value": v})

    async def get(self) -> dict:
        return await self._rpc({"cmd": "GET"})

    async def ping(self) -> dict:
        return await self._rpc({"cmd": "PING"})


# ---------- Optional sync-friendly wrapper ----------
_loop = None
_thread = None

def _ensure_loop():
    global _loop, _thread
    if _loop and _loop.is_running():
        return _loop

    _loop = asyncio.new_event_loop()
    _thread = threading.Thread(target=_loop.run_forever, daemon=True)
    _thread.start()
    return _loop

def _run_coro(coro):
    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()

class GripperClient:
    """Sync convenience wrapper. Prefer GripperClientAsync in async code."""
    def __init__(self, host: str = "127.0.0.1", port: int = 22345, read_timeout: float = 5.0):
        self._c = GripperClientAsync(host, port, read_timeout)

    def set(self, v: float):
        return _run_coro(self._c.set(v))

    def get(self):
        return _run_coro(self._c.get())

    def ping(self):
        return _run_coro(self._c.ping())

    def disconnect(self):
        return _run_coro(self._c.disconnect())


# Quick test
if __name__ == "__main__":
    async def main():
        cli = GripperClientAsync()
        print(await cli.ping())
        print(await cli.set(0.6))
        print(await cli.get())
        await cli.disconnect()
    asyncio.run(main())
