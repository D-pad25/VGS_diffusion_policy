#!/usr/bin/env python3
import asyncio
import json

class MockGripperServer:
    def __init__(self, host='127.0.0.1', port=22345):
        self.host = host
        self.port = port
        self.position = 0.0  # Default gripper state

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        print(f"[🟢] Client connected: {addr}")

        try:
            while True:
                data = await reader.readline()
                if not data:
                    break

                try:
                    message = json.loads(data.decode())
                    cmd = message.get("cmd")
                    value = message.get("value")

                    if cmd == "SET":
                        val = float(value)
                        self.position = max(0.0, min(1.0, val))
                        response = {"status": "OK", "value": round(self.position, 3)}
                    elif cmd == "GET":
                        response = {"position": round(self.position, 3)}
                    else:
                        response = {"error": f"Unknown command '{cmd}'"}

                except Exception as e:
                    response = {"error": str(e)}

                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

        except Exception as e:
            print(f"[⚠️] Error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"[🔌] Client disconnected: {addr}")

    async def start(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        print(f"[🔌] MockGripperServer running on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(MockGripperServer().start())
