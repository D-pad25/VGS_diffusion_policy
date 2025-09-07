#!/usr/bin/env python3
"""
Async JSON line protocol server bridging TCP <-> ROS gripper topics.

Commands (newline-delimited JSON):
  {"cmd":"SET","value":0.0..1.0}  -> publishes Int16 [0..255] to /gripper_command
  {"cmd":"GET"}                   -> returns latest measured position (0..1)
  {"cmd":"PING"}                  -> health check
"""

import asyncio
import json
import argparse
import threading
import time
import rospy
from std_msgs.msg import Int16, Float32


def clamp01(x: float) -> float:
    x = float(x)
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x


class GripperSocketServerAsync:
    def __init__(self, host: str, port: int, idle_timeout: float = 15.0,
                 topic_cmd: str = "/gripper_command",
                 topic_cmd_norm: str = "/gripper_command_norm",
                 topic_pos: str = "/gripper_position"):
        self.host = host
        self.port = port
        self.idle_timeout = idle_timeout

        # ROS pubs/subs
        self._pub_cmd = rospy.Publisher(topic_cmd, Int16, queue_size=1)        # command 0..255
        self._pub_cmd_norm = rospy.Publisher(topic_cmd_norm, Float32, queue_size=1)  # command 0..1 (debug)
        self._topic_cmd = topic_cmd

        self._pos_lock = threading.Lock()
        self._latest_pos_norm = None  # Float in [0..1], measured
        self._last_cmd_norm = None    # Float in [0..1], commanded

        rospy.Subscriber(topic_pos, Float32, self._gripper_cb)

        # state for first-connection wait
        self._pub_ready = False
        self._waited_once = False

    # -------------------- helpers --------------------
    def _wait_for_subscribers(self, timeout_s: float = 3.0) -> bool:
        """Block briefly until at least one subscriber is connected (or timeout)."""
        if self._pub_ready:
            return True
        deadline = time.time() + timeout_s
        while time.time() < deadline and not rospy.is_shutdown():
            conns = self._pub_cmd.get_num_connections()
            if conns > 0:
                rospy.loginfo(f"[ROS] {self._topic_cmd} has {conns} subscriber(s).")
                self._pub_ready = True
                return True
            time.sleep(0.05)
        if not self._waited_once:
            rospy.logwarn(f"[ROS] No subscribers on {self._topic_cmd} after {timeout_s:.1f}s; commands may be dropped.")
            self._waited_once = True
        return False

    # -------------------- ROS callback --------------------
    def _gripper_cb(self, msg: Float32):
        norm = float(msg.data) / 255.0
        self._latest_pos_norm = clamp01(norm)

    # -------------------- TCP handling --------------------
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        print(f"[🟢] Client connected: {addr}")
        try:
            while not rospy.is_shutdown():
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=self.idle_timeout)
                except asyncio.TimeoutError:
                    print(f"[⏱] Idle timeout from {addr}; closing.")
                    break
                if not line:
                    break

                reply = self._dispatch_line(line)
                writer.write((json.dumps(reply, separators=(',', ':')) + "\n").encode())
                await writer.drain()

        except Exception as e:
            print(f"[⚠️] Client error {addr}: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            print(f"[🔌] Client disconnected: {addr}")

    def _dispatch_line(self, raw: bytes) -> dict:
        try:
            msg = json.loads(raw.decode().strip() or "{}")
            cmd = str(msg.get("cmd", "")).upper()
        except Exception as e:
            return {"ok": False, "error": f"invalid json: {e}"}

        if cmd == "SET":
            if "value" not in msg:
                return {"ok": False, "error": "SET requires 'value' in [0,1]"}
            try:
                val = clamp01(float(msg["value"]))
            except Exception:
                return {"ok": False, "error": "SET 'value' must be numeric in [0,1]"}
            return self._do_set(val)

        if cmd == "GET":
            with self._pos_lock:
                pos = self._latest_pos_norm
                last_cmd = self._last_cmd_norm
            return {
                "ok": True,
                "cmd": "GET",
                "position": pos if pos is not None else None,
                "last_cmd": last_cmd,
                "units": "0..1",
                "timestamp": rospy.get_time(),
            }

        if cmd == "PING":
            return {"ok": True, "cmd": "PING", "timestamp": rospy.get_time()}

        return {"ok": False, "error": f"unknown cmd '{cmd}'"}

    def _do_set(self, value_norm: float) -> dict:
        # Ensure someone is listening (best-effort). If not, we still publish.
        self._wait_for_subscribers(timeout_s=3.0)

        int_val = int(round(value_norm * 255))
        # Publish both raw int command and normalized debug
        self._pub_cmd.publish(Int16(data=int_val))
        self._pub_cmd_norm.publish(Float32(data=value_norm))
        with self._pos_lock:
            self._last_cmd_norm = value_norm

        rospy.loginfo(f"[➡️ SET] norm={value_norm:.3f} -> int16={int_val} (subs={self._pub_cmd.get_num_connections()})")
        return {
            "ok": True,
            "cmd": "SET",
            "value": value_norm,
            "int_value": int_val,
            "timestamp": rospy.get_time(),
        }

    # -------------------- server lifecycle --------------------
    async def start(self):
        server = await asyncio.start_server(self._handle_client, self.host, self.port)
        print(f"[🔌] Gripper Async Server running on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=22345)
    parser.add_argument("--idle-timeout", type=float, default=15.0)
    parser.add_argument("--topic-cmd", default="/gripper_command")
    parser.add_argument("--topic-cmd-norm", default="/gripper_command_norm")
    parser.add_argument("--topic-pos", default="/gripper_position")
    args = parser.parse_args()

    rospy.init_node("gripper_socket_server_async", anonymous=True, disable_signals=True)

    srv = GripperSocketServerAsync(args.host, args.port, args.idle_timeout,
                                   topic_cmd=args.topic_cmd,
                                   topic_cmd_norm=args.topic_cmd_norm,
                                   topic_pos=args.topic_pos)
    try:
        asyncio.run(srv.start())
    except KeyboardInterrupt:
        print("\n[⏹] Server shutdown requested.")
    finally:
        try:
            rospy.signal_shutdown("server exit")
        except Exception:
            pass


if __name__ == "__main__":
    main()
