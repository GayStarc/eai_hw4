import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from typing import Optional, Set
from .data_models import HumanoidData, CameraData
from queue import Queue

class RobotServer:
    def __init__(self):
        self._humanoid_clients: Set = set()
        self._quadruped_clients: Set = set()
        self._humanoid_message_queue = Queue()
        self._quadruped_message_queue = Queue()
        self._latest_data: Optional[HumanoidData] = None
        self._data_lock = asyncio.Lock()
        self._server = None
        self._loop = None

    def start(self, host="0.0.0.0", port=8765):
        """Start the WebSocket server"""
        async def _start(host, port):
            async with websockets.serve(self._handle_connection, host, port,
                max_size=10*1024*1024):
                print(f"Server running at ws://{host}:{port}")
                await asyncio.gather(
                    self._broadcast_humanoid_messages(),
                    self._broadcast_quadruped_messages()
                )

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(_start(host, port))

    async def _broadcast_humanoid_messages(self):
        while True:
            if not self._humanoid_message_queue.empty():
                message = self._humanoid_message_queue.get()
                print(f"Sending to humanoids: {message}")
                async with self._data_lock:
                    if self._humanoid_clients:
                        await asyncio.gather(
                            *[ws.send(message) for ws in self._humanoid_clients],
                            return_exceptions=True
                        )
            await asyncio.sleep(0.1)

    async def _broadcast_quadruped_messages(self):
        while True:
            if not self._quadruped_message_queue.empty():
                message = self._quadruped_message_queue.get()
                print(f"Sending to quadrupeds: {message}")
                async with self._data_lock:
                    if self._quadruped_clients:
                        await asyncio.gather(
                            *[ws.send(message) for ws in self._quadruped_clients],
                            return_exceptions=True
                        )
            await asyncio.sleep(0.1)

    def check_active_connections(self):
        """Check if any clients are connected"""
        return len(self._humanoid_clients) > 0 or len(self._quadruped_clients) > 0

    async def stop(self):
        """Stop the server"""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def get_latest_data(self) -> Optional[HumanoidData]:
        """Get the latest sensor data"""
        # async with self._data_lock:
        return self._latest_data

    def send_humanoid_command(self, trajectory: list):
        """Send humanoid robot control command"""
        if not self._humanoid_clients:
            raise ConnectionError("No humanoid connected")
        
        message = json.dumps({
            "type": "humanoid_control",
            "trajectory": trajectory
        })
        self._humanoid_message_queue.put(message)
        
    def send_humanoid_gripper(self, gripper_value: int):
        """Send humanoid robot gripper command"""
        if not self._humanoid_clients:
            raise ConnectionError("No humanoid connected")
        
        message = json.dumps({
            "type": "gripper_control",
            "gripper_value": gripper_value
        })
        self._humanoid_message_queue.put(message)
        
    def send_humanoid_head(self, head_value: list):
        """Send humanoid robot head command"""
        if not self._humanoid_clients:
            raise ConnectionError("No humanoid connected")
        
        message = json.dumps({
            "type": "head_control",
            "head_value": head_value
        })
        self._humanoid_message_queue.put(message)
        
    def send_humanoid_leg(self, leg_value: list):
        """Send humanoid robot leg command"""
        if not self._humanoid_clients:
            raise ConnectionError("No humanoid connected")
        
        message = json.dumps({
            "type": "leg_control",
            "leg_value": leg_value
        })
        self._humanoid_message_queue.put(message)

    def send_quadruped_command(self, x_speed: float, y_speed: float, yaw_speed: float):
        """Send quadruped robot speed command"""
        if not self._quadruped_clients:
            raise ConnectionError("No quadruped connected")
        
        message = json.dumps({
            "type": "quadruped_control",
            "x_speed": x_speed,
            "y_speed": y_speed,
            "yaw_speed": yaw_speed
        })
        self._quadruped_message_queue.put(message)

    async def _handle_connection(self, websocket, path):
        """Handle client connection"""
        try:
            print(f"New connection on path: {path}")
            if "humanoid" in path:
                await self._handle_humanoid(websocket)
            elif "quadruped" in path:
                await self._handle_quadruped(websocket)
            else:
                print(f"Unknown path: {path}")
                await websocket.close()
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            await self._cleanup(websocket)

    async def _handle_humanoid(self, websocket):
        """Handle humanoid robot data"""
        async with self._data_lock:
            self._humanoid_clients.add(websocket)
            print(f"New humanoid connected. Total humanoids: {len(self._humanoid_clients)}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                # if len(data) > 200:
                #     print(f"Received humanoid data: {data[:200]}")
                # else:
                #     print(f"Received humanoid data: {data}")
                if data['type'] == 'humanoid_data':
                    await self._update_data(data)
        except websockets.exceptions.ConnectionClosed:
            print("Humanoid connection closed")
        finally:
            async with self._data_lock:
                if websocket in self._humanoid_clients:
                    self._humanoid_clients.remove(websocket)
                    print(f"Humanoid disconnected. Total humanoids: {len(self._humanoid_clients)}")

    async def _handle_quadruped(self, websocket):
        """Handle quadruped robot connection"""
        async with self._data_lock:
            self._quadruped_clients.add(websocket)
            print(f"New quadruped connected. Total quadrupeds: {len(self._quadruped_clients)}")
        
        try:
            while True:  # Keep the connection alive
                message = await websocket.recv()
                print(f"Received quadruped message: {message}")
        except websockets.exceptions.ConnectionClosed:
            print("Quadruped connection closed")
        finally:
            async with self._data_lock:
                if websocket in self._quadruped_clients:
                    self._quadruped_clients.remove(websocket)
                    print(f"Quadruped disconnected. Total quadrupeds: {len(self._quadruped_clients)}")

    async def _update_data(self, raw_data: dict):
        """Update sensor data"""
        humanoid_data = HumanoidData(
            head_camera=self._parse_camera(raw_data['head_camera']),
            wrist_camera=self._parse_camera(raw_data['wrist_camera']),
            joint_angles=raw_data['joint_angles'],
            timestamp=raw_data['timestamp']
        )
        async with self._data_lock:
            self._latest_data = humanoid_data

    def _decode_image(self, b64_str: str) -> np.ndarray:
        data = base64.b64decode(b64_str)
        img_array = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

    def _parse_camera(self, camera_data: dict) -> CameraData:
        """Parse camera data"""
        return CameraData(
            color_image=self._decode_image(camera_data['color']),
            depth_image=self._decode_image(camera_data['depth']),
            intrinsics=camera_data['intrinsic'],
            pose=camera_data['pose']
        )

    async def _cleanup(self, websocket):
        """Clean up on disconnect"""
        async with self._data_lock:
            if websocket in self._humanoid_clients:
                self._humanoid_clients.remove(websocket)
                print(f"Humanoid disconnected. Total humanoids: {len(self._humanoid_clients)}")
            if websocket in self._quadruped_clients:
                self._quadruped_clients.remove(websocket)
                print(f"Quadruped disconnected. Total quadrupeds: {len(self._quadruped_clients)}")