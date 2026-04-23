"""
pi5_controller.py
─────────────────────────────────────────────────────────────────────────────
Raspberry Pi 5  →  Pico controller via Serial1 (physical UART, /dev/ttyAMA0)
─────────────────────────────────────────────────────────────────────────────

Wiring (Pi 5  ↔  Pico):
  Pi5  GPIO14 (TX, pin 8)  →  Pico GP1 (RX, pin 2)
  Pi5  GPIO15 (RX, pin 10) ←  Pico GP0 (TX, pin 1)
  GND  (pin 6/14/…)        ↔  GND (pin 3/38)

Install deps:
  pip install pyserial psutil

Enable UART on Pi 5 (/boot/firmware/config.txt):
  enable_uart=1
  dtoverlay=disable-bt        # frees /dev/ttyAMA0 from Bluetooth

Usage example:
    from pi5_controller import PicoController
    with PicoController('/dev/ttyAMA0') as ctl:
        ctl.ping()
        ctl.mouse_move_rel(10, -5)
        ctl.kb_write("hello world")
"""

import struct
import time
import threading
import logging
from typing import Optional

try:
    import serial
except ImportError:
    raise ImportError("pyserial not found – run: pip install pyserial")

try:
    import psutil
except ImportError:
    psutil = None  # optional; needed for agent sysinfo commands

# ─── Protocol constants ───────────────────────────────────────────────────────
ACK_BYTE   = 0x06
NAK_BYTE   = 0x15
RESP_BYTE  = 0x02   # STX – marks start of a response from Pico
END_BYTE   = 0xFF
HDD_EVENT  = 0xE1   # async HDD-LED event from Pico

# ─── Command codes ────────────────────────────────────────────────────────────
CMD_PING               = 0
CMD_MOUSE_MOVE_REL     = 1
CMD_MOUSE_MOVE_ABS     = 2
CMD_MOUSE_CLICK        = 3
CMD_MOUSE_HOLD         = 4
CMD_MOUSE_RELEASE      = 5
CMD_MOUSE_RELEASE_ALL  = 6
CMD_KEY_TYPE           = 7
CMD_KEY_HOLD           = 8
CMD_KEY_RELEASE        = 9
CMD_KEY_RELEASE_ALL    = 10
CMD_MOUSE_SCROLL       = 11
CMD_KB_WRITE           = 12
CMD_CTRL_ALT_DEL       = 13
CMD_CTRL_ALT_FX        = 14
CMD_CTRL_ALT_ESC       = 15
CMD_KB_COMBO           = 16
CMD_MAGIC_SYSRQ        = 17
CMD_CTRL_ALT_BKSP      = 18
CMD_SYSINFO            = 19
CMD_CPU_NOW            = 20
CMD_MEM_NOW            = 21
CMD_SPACE_NOW          = 22
CMD_NET_NOW            = 23
CMD_UPTIME_NOW         = 24
CMD_AGENT_PING         = 25
CMD_AGENT_VERSION      = 26
CMD_POWER_PRESS        = 27
CMD_POWER_HOLD         = 28
CMD_POWER_RELEASE      = 29
CMD_RESET_PRESS        = 30
CMD_GET_POWER_LED      = 31
CMD_HDD_LED_REPORT     = 32
CMD_AGENT_CMD          = 33  # Send custom command string to agent

# HID modifier keys (same as Arduino Keyboard.h)
KEY_LEFT_CTRL   = 0x80
KEY_LEFT_SHIFT  = 0x81
KEY_LEFT_ALT    = 0x82
KEY_LEFT_GUI    = 0x83
KEY_RIGHT_CTRL  = 0x84
KEY_RIGHT_SHIFT = 0x85
KEY_RIGHT_ALT   = 0x86
KEY_RIGHT_GUI   = 0x87

# Common HID key codes
KEY_RETURN      = 0xB0
KEY_ESC         = 0xB1
KEY_BACKSPACE   = 0xB2
KEY_TAB         = 0xB3
KEY_CAPS_LOCK   = 0xC1
KEY_DELETE      = 0xD4
KEY_INSERT      = 0xD1
KEY_HOME        = 0xD2
KEY_END         = 0xD5
KEY_PAGE_UP     = 0xD3
KEY_PAGE_DOWN   = 0xD6
KEY_UP_ARROW    = 0xDA
KEY_DOWN_ARROW  = 0xD9
KEY_LEFT_ARROW  = 0xD8
KEY_RIGHT_ARROW = 0xD7
KEY_F1          = 0xC2
KEY_F2          = 0xC3
KEY_F3          = 0xC4
KEY_F4          = 0xC5
KEY_F5          = 0xC6
KEY_F6          = 0xC7
KEY_F7          = 0xC8
KEY_F8          = 0xC9
KEY_F9          = 0xCA
KEY_F10         = 0xCB
KEY_F11         = 0xCC
KEY_F12         = 0xCD

MOUSE_LEFT   = 0
MOUSE_MIDDLE = 1
MOUSE_RIGHT  = 2

SYSRQ_R = 0; SYSRQ_E = 1; SYSRQ_I = 2; SYSRQ_S = 3
SYSRQ_U = 4; SYSRQ_B = 5; SYSRQ_O = 6

log = logging.getLogger("PicoController")


# ─── Exceptions ───────────────────────────────────────────────────────────────
class PicoNAK(Exception):
    """Pico returned NAK (bad/missing args or timeout on its side)."""

class PicoTimeout(Exception):
    """No response from Pico within the timeout window."""

class PicoProtocolError(Exception):
    """Unexpected byte in response stream."""


# ─── Controller class ─────────────────────────────────────────────────────────
class PicoController:
    """
    High-level Python API for the Pico KVM agent.

    Thread safety: a single internal lock serialises all send/recv pairs,
    so you can safely call methods from multiple threads.
    """

    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baud: int = 115200,
        timeout: float = 2.0,
        hdd_led_callback=None,
    ):
        self._port    = port
        self._baud    = baud
        self._timeout = timeout
        self._lock    = threading.Lock()
        self._ser: Optional[serial.Serial] = None

        # Optional callback(state: bool) for async HDD LED events
        self._hdd_callback = hdd_led_callback
        self._reader_thread: Optional[threading.Thread] = None

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ── Connection ───────────────────────────────────────────────────────────

    def connect(self):
        self._ser = serial.Serial(self._port, self._baud, timeout=self._timeout)
        time.sleep(0.1)  # let UART settle
        log.info(f"Connected to Pico on {self._port} @ {self._baud}")

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
        log.info("Disconnected from Pico")

    # ── Low-level I/O ────────────────────────────────────────────────────────

    def _send(self, data: bytes):
        self._ser.write(data)

    def _recv_byte(self) -> int:
        b = self._ser.read(1)
        if not b:
            raise PicoTimeout("Timed out waiting for byte from Pico")
        return b[0]

    def _read_response(self) -> bytes:
        """
        Read a Pico response:
          ACK  → return b''
          NAK  → raise PicoNAK
          RESP → read length byte, then read that many bytes, return payload
        """
        b = self._recv_byte()

        if b == ACK_BYTE:
            return b""

        if b == NAK_BYTE:
            raise PicoNAK("Pico returned NAK")

        if b == RESP_BYTE:
            length = self._recv_byte()
            if length == 0:
                return b""
            payload = self._ser.read(length)
            if len(payload) < length:
                raise PicoTimeout(f"Short read: got {len(payload)}/{length} bytes")
            return payload

        raise PicoProtocolError(f"Unexpected leading byte: 0x{b:02X}")

    def _cmd(self, data: bytes) -> bytes:
        """Send command bytes and return the response payload (thread-safe)."""
        with self._lock:
            self._ser.reset_input_buffer()
            self._send(data)
            return self._read_response()

    # ── CMD 0: Ping ──────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Returns True if Pico responds with PONG."""
        resp = self._cmd(bytes([CMD_PING]))
        return len(resp) > 0 and resp[0] == 0x50

    # ── CMD 1: Mouse move (relative) ─────────────────────────────────────────

    def mouse_move_rel(self, dx: int, dy: int):
        """Move mouse by (dx, dy). Values clamped to [-128, 127]."""
        dx = max(-128, min(127, dx))
        dy = max(-128, min(127, dy))
        self._cmd(bytes([CMD_MOUSE_MOVE_REL, dx & 0xFF, dy & 0xFF]))

    # ── CMD 2: Mouse move (absolute) ─────────────────────────────────────────

    def mouse_move_abs(self, x: int, y: int):
        """Move mouse to absolute position (x, y). Range 0–32767."""
        x = max(0, min(32767, x))
        y = max(0, min(32767, y))
        payload = struct.pack(">HH", x, y)
        self._cmd(bytes([CMD_MOUSE_MOVE_ABS]) + payload)

    # ── CMD 3: Mouse click ───────────────────────────────────────────────────

    def mouse_click(self, button: int = MOUSE_LEFT):
        self._cmd(bytes([CMD_MOUSE_CLICK, button]))

    # ── CMD 4: Mouse hold ────────────────────────────────────────────────────

    def mouse_hold(self, button: int = MOUSE_LEFT):
        self._cmd(bytes([CMD_MOUSE_HOLD, button]))

    # ── CMD 5: Mouse release ─────────────────────────────────────────────────

    def mouse_release(self, button: int = MOUSE_LEFT):
        self._cmd(bytes([CMD_MOUSE_RELEASE, button]))

    # ── CMD 6: Mouse release all ─────────────────────────────────────────────

    def mouse_release_all(self):
        self._cmd(bytes([CMD_MOUSE_RELEASE_ALL]))

    # ── CMD 7: Key type (press+release) ──────────────────────────────────────

    def key_type(self, key: int):
        """Press and release a single key (HID keycode)."""
        self._cmd(bytes([CMD_KEY_TYPE, key & 0xFF]))

    # ── CMD 8: Key hold ──────────────────────────────────────────────────────

    def key_hold(self, key: int):
        self._cmd(bytes([CMD_KEY_HOLD, key & 0xFF]))

    # ── CMD 9: Key release ───────────────────────────────────────────────────

    def key_release(self, key: int):
        self._cmd(bytes([CMD_KEY_RELEASE, key & 0xFF]))

    # ── CMD 10: Key release all ──────────────────────────────────────────────

    def key_release_all(self):
        self._cmd(bytes([CMD_KEY_RELEASE_ALL]))

    # ── CMD 11: Mouse scroll ─────────────────────────────────────────────────

    def mouse_scroll(self, delta: int):
        """Scroll wheel. Positive = up, negative = down. Clamped to [-128,127]."""
        delta = max(-128, min(127, delta))
        self._cmd(bytes([CMD_MOUSE_SCROLL, delta & 0xFF]))

    # ── CMD 12: Keyboard write (text string) ─────────────────────────────────

    def kb_write(self, text: str, encoding: str = "ascii"):
        """Type a full string. Non-ASCII chars are skipped if encoding fails."""
        encoded = text.encode(encoding, errors="ignore")
        payload = bytes([CMD_KB_WRITE]) + encoded + bytes([END_BYTE])
        self._cmd(payload)

    # ── CMD 13: Ctrl+Alt+Del ─────────────────────────────────────────────────

    def ctrl_alt_del(self):
        self._cmd(bytes([CMD_CTRL_ALT_DEL]))

    # ── CMD 14: Ctrl+Alt+Fx ──────────────────────────────────────────────────

    def ctrl_alt_fx(self, fx: int):
        """fx in 1–12 for F1–F12."""
        if not 1 <= fx <= 12:
            raise ValueError("fx must be 1–12")
        self._cmd(bytes([CMD_CTRL_ALT_FX, fx]))

    # ── CMD 15: Ctrl+Alt+Esc ─────────────────────────────────────────────────

    def ctrl_alt_esc(self):
        self._cmd(bytes([CMD_CTRL_ALT_ESC]))

    # ── CMD 16: Keyboard combo ───────────────────────────────────────────────

    def kb_combo(self, modifier: int, key: int):
        """
        Press modifier + key then release all.
        E.g. kb_combo(KEY_LEFT_GUI, ord('r'))  → Win+R
        """
        self._cmd(bytes([CMD_KB_COMBO, modifier & 0xFF, key & 0xFF]))

    # ── CMD 17: Magic SysRq ──────────────────────────────────────────────────

    def magic_sysrq(self, key_index: int):
        """
        key_index: SYSRQ_R=0, SYSRQ_E=1, SYSRQ_I=2, SYSRQ_S=3,
                   SYSRQ_U=4, SYSRQ_B=5, SYSRQ_O=6
        """
        self._cmd(bytes([CMD_MAGIC_SYSRQ, key_index & 0xFF]))

    # ── CMD 18: Ctrl+Alt+Backspace ───────────────────────────────────────────

    def ctrl_alt_backspace(self):
        self._cmd(bytes([CMD_CTRL_ALT_BKSP]))

    # ── CMDs 19–24: Agent system info (Pi 5 responds to Pico's forwarded req) ─

    def sysinfo(self) -> str:
        resp = self._cmd(bytes([CMD_SYSINFO]))
        return resp.decode("utf-8", errors="replace")

    def cpu_now(self) -> str:
        resp = self._cmd(bytes([CMD_CPU_NOW]))
        return resp.decode("utf-8", errors="replace")

    def memory_now(self) -> str:
        resp = self._cmd(bytes([CMD_MEM_NOW]))
        return resp.decode("utf-8", errors="replace")

    def space_now(self) -> str:
        resp = self._cmd(bytes([CMD_SPACE_NOW]))
        return resp.decode("utf-8", errors="replace")

    def network_now(self) -> str:
        resp = self._cmd(bytes([CMD_NET_NOW]))
        return resp.decode("utf-8", errors="replace")

    def uptime_now(self) -> str:
        resp = self._cmd(bytes([CMD_UPTIME_NOW]))
        return resp.decode("utf-8", errors="replace")

    # ── CMD 25: Agent ping ───────────────────────────────────────────────────

    def agent_ping(self) -> bool:
        resp = self._cmd(bytes([CMD_AGENT_PING]))
        return len(resp) > 0 and resp[0] == 0x50

    # ── CMD 26: Agent version ────────────────────────────────────────────────

    def agent_version(self) -> str:
        resp = self._cmd(bytes([CMD_AGENT_VERSION]))
        if len(resp) >= 3:
            return f"{resp[0]}.{resp[1]}.{resp[2]}"
        return "unknown"

    # ── CMD 27–29: Power button ──────────────────────────────────────────────

    def power_press(self):
        """Short power button press (200 ms pulse)."""
        self._cmd(bytes([CMD_POWER_PRESS]))

    def power_hold(self):
        """Hold power button (stays HIGH until power_release)."""
        self._cmd(bytes([CMD_POWER_HOLD]))

    def power_release(self):
        """Release power button."""
        self._cmd(bytes([CMD_POWER_RELEASE]))

    def power_long_press(self, hold_secs: float = 5.0):
        """Hold power for hold_secs then release (force shutdown pattern)."""
        self.power_hold()
        time.sleep(hold_secs)
        self.power_release()

    # ── CMD 30: Reset button ─────────────────────────────────────────────────

    def reset_press(self):
        """Short reset button pulse."""
        self._cmd(bytes([CMD_RESET_PRESS]))

    # ── CMD 31: Get power LED state ──────────────────────────────────────────

    def get_power_led(self) -> bool:
        resp = self._cmd(bytes([CMD_GET_POWER_LED]))
        return len(resp) > 0 and resp[0] == 1

    # ── CMD 32: HDD LED interrupt report ─────────────────────────────────────

    def set_hdd_led_report(self, enable: bool = True):
        self._cmd(bytes([CMD_HDD_LED_REPORT, 1 if enable else 0]))

    # ── CMD 33: Agent custom command ─────────────────────────────────────────

    def agent_cmd(self, command: str, args: str = "") -> str:
        """
        Send a custom command string directly to the Pi 5 agent.

        The agent receives: "<command> <args>" (space-separated if args given).
        Returns the agent's response as a string.

        Examples:
            ctl.agent_cmd("screenshot")
            ctl.agent_cmd("shell", "uptime")
            ctl.agent_cmd("notify", "title=Hello body=World")
            ctl.agent_cmd("myplug", "foo=bar")
        """
        full = command if not args else f"{command} {args}"
        payload = full.encode("utf-8", errors="ignore")
        resp = self._cmd(bytes([CMD_AGENT_CMD]) + payload + bytes([END_BYTE]))
        return resp.decode("utf-8", errors="replace")

    # ── Convenience helpers ──────────────────────────────────────────────────

    def type_enter(self):
        self.key_type(KEY_RETURN)

    def type_tab(self):
        self.key_type(KEY_TAB)

    def type_escape(self):
        self.key_type(KEY_ESC)

    def win_r(self):
        """Open the Windows Run dialog."""
        self.kb_combo(KEY_LEFT_GUI, ord('r'))

    def copy(self):
        self.kb_combo(KEY_LEFT_CTRL, ord('c'))

    def paste(self):
        self.kb_combo(KEY_LEFT_CTRL, ord('v'))

    def select_all(self):
        self.kb_combo(KEY_LEFT_CTRL, ord('a'))

    def mouse_drag(self, x1, y1, x2, y2, steps: int = 20, delay: float = 0.01):
        """Absolute drag from (x1,y1) to (x2,y2)."""
        self.mouse_move_abs(x1, y1)
        time.sleep(0.05)
        self.mouse_hold(MOUSE_LEFT)
        for i in range(1, steps + 1):
            ix = x1 + int((x2 - x1) * i / steps)
            iy = y1 + int((y2 - y1) * i / steps)
            self.mouse_move_abs(ix, iy)
            time.sleep(delay)
        self.mouse_release(MOUSE_LEFT)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pico KVM controller CLI test")
    parser.add_argument("--port",  default="/dev/ttyAMA0",  help="UART port")
    parser.add_argument("--cdc",   default="/dev/ttyACM0",  help="USB CDC port for agent daemon")
    parser.add_argument("--baud",  default=115200, type=int)
    parser.add_argument("--cmd",   default="ping",           help="Command to run")
    parser.add_argument("--text",  default="hello",          help="Text for kb_write")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with PicoController(port=args.port, baud=args.baud) as ctl:
        match args.cmd:
            case "ping":
                print("PONG!" if ctl.ping() else "No response")
            case "agent_ping":
                print("Agent PONG!" if ctl.agent_ping() else "No agent response")
            case "version":
                print(f"Firmware version: {ctl.agent_version()}")
            case "type":
                ctl.kb_write(args.text)
                print(f"Typed: {args.text!r}")
            case "cad":
                ctl.ctrl_alt_del()
                print("Sent Ctrl+Alt+Del")
            case "power":
                ctl.power_press()
                print("Power button pressed")
            case "reset":
                ctl.reset_press()
                print("Reset button pressed")
            case "led":
                print(f"Power LED: {'ON' if ctl.get_power_led() else 'OFF'}")
            case "sysinfo":
                print(ctl.sysinfo())
            case "cpu":
                print(f"CPU: {ctl.cpu_now()}")
            case "mem":
                print(f"Memory: {ctl.memory_now()}")
            case "space":
                print(f"Disk: {ctl.space_now()}")
            case "net":
                print(f"Network: {ctl.network_now()}")
            case "uptime":
                print(f"Uptime: {ctl.uptime_now()}")
            case _:
                print(f"Unknown command: {args.cmd}")
