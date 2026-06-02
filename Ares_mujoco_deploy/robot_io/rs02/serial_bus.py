from __future__ import annotations

import threading
import time

try:
    import can
except ImportError as exc:
    can = None
    CAN_IMPORT_ERROR = exc
else:
    CAN_IMPORT_ERROR = None

try:
    import serial
except ImportError as exc:
    serial = None
    SERIAL_IMPORT_ERROR = exc
else:
    SERIAL_IMPORT_ERROR = None


class RS02AtSerialBus:
    """RobStride CH340 AT-mode USB-CAN adapter."""

    HEADER = b"AT"
    TAIL = b"\r\n"

    def __init__(self, channel: str, tty_baudrate: int = 921600, timeout: float = 0.02):
        if serial is None:
            raise RuntimeError(f"pyserial is not installed: {SERIAL_IMPORT_ERROR}")
        if can is None:
            raise RuntimeError(f"python-can is not installed: {CAN_IMPORT_ERROR}")

        self.serial = serial.Serial(
            port=channel,
            baudrate=tty_baudrate,
            timeout=timeout,
            write_timeout=0.2,
        )
        self._rx_buffer = bytearray()
        self._io_lock = threading.Lock()

    def send(self, msg: "can.Message", timeout: float | None = None) -> None:
        payload = bytes(msg.data)
        if len(payload) > 8:
            raise ValueError("CAN data field cannot exceed 8 bytes")

        encoded_id = (((msg.arbitration_id & 0x1FFFFFFF) << 3) | 0x4) & 0xFFFFFFFF
        frame = self.HEADER + encoded_id.to_bytes(4, "big") + bytes([len(payload)]) + payload + self.TAIL
        with self._io_lock:
            self.serial.write(frame)
            self.serial.flush()

    def recv(self, timeout: float | None = None) -> "can.Message | None":
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            msg = self._try_parse_frame()
            if msg is not None:
                return msg

            if deadline is not None and time.monotonic() >= deadline:
                return None

            waiting = self.serial.in_waiting
            chunk = self.serial.read(waiting if waiting else 1)
            if chunk:
                self._rx_buffer.extend(chunk)
            elif timeout is not None:
                time.sleep(0.001)

    def shutdown(self) -> None:
        self.serial.close()

    def _try_parse_frame(self) -> "can.Message | None":
        header_index = self._rx_buffer.find(self.HEADER)
        if header_index < 0:
            self._rx_buffer.clear()
            return None
        if header_index > 0:
            del self._rx_buffer[:header_index]

        minimum_len = len(self.HEADER) + 4 + 1 + len(self.TAIL)
        if len(self._rx_buffer) < minimum_len:
            return None

        dlc = self._rx_buffer[6]
        if dlc > 8:
            del self._rx_buffer[0]
            return None

        frame_len = len(self.HEADER) + 4 + 1 + dlc + len(self.TAIL)
        if len(self._rx_buffer) < frame_len:
            return None

        frame = bytes(self._rx_buffer[:frame_len])
        del self._rx_buffer[:frame_len]
        if not frame.endswith(self.TAIL):
            return None

        encoded_id = int.from_bytes(frame[2:6], "big")
        arbitration_id = (encoded_id >> 3) & 0x1FFFFFFF
        return can.Message(
            arbitration_id=arbitration_id,
            data=frame[7 : 7 + dlc],
            is_extended_id=True,
            is_remote_frame=False,
        )
