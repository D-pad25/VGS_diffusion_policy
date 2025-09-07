import asyncio
import json

class TestGripperClient:
    def __init__(self, host='127.0.0.1', port=22345):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        print("[Client] Connected.")

    async def disconnect(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            print("[Client] Disconnected.")

    async def send_command(self, cmd, value=None):
        payload = {"cmd": cmd, "value": value}
        self.writer.write((json.dumps(payload) + "\n").encode())
        await self.writer.drain()
        response = await self.reader.readline()
        print(f"[Client] Server response: {response.decode().strip()}")

    async def run(self):
        await self.connect()

        print("\n--- TESTING MOCK GRIPPER ---")

        # Test 1: Set gripper to 0.8
        await self.send_command("SET", 0.8)

        # Test 2: Get current position
        await self.send_command("GET")

        # Test 3: Send invalid command
        await self.send_command("PING")

        await self.disconnect()


if __name__ == '__main__':
    asyncio.run(TestGripperClient().run())
