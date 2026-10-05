"""Local DCSS 0.17.1 WebTiles transport, with a strict observable allowlist.

Uses the game's native Unix datagram protocol; no web server or public account.
Constants/protocol verified against crawl/crawl tag 0.17.1 (see docs/DCSS_LOOP.md).
"""
import copy
from collections import deque
import os
import select
import json
import socket
import time

UNSEEN = 0x40000
MM_UNSEEN = 0x20000  # magically mapped/detected but not currently visible
COMMAND_MODE = 1
NORMAL_UI = 0
ACTIONS = {"north": "k", "east": "l", "south": "j", "west": "h", "wait": "."}
DELTAS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0), "wait": (0, 0)}


class UnsafeInput(RuntimeError):
    pass


class MessageDecoder:
    """Reassemble datagram fragments. '*' prefixes server-internal messages."""
    def __init__(self):
        self.buffer = b""

    def feed(self, data):
        self.buffer += data
        if len(self.buffer) > 8 * 1024 * 1024:
            raise ValueError("oversized protocol message")
        result = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            if line:
                result.append(json.loads(line.lstrip(b"*")))
        return result


def tile_bits(value):
    if isinstance(value, list):
        return (int(value[0]) & 0xffffffff) | (int(value[1]) << 32)
    return int(value) & 0xffffffff


class ObservationState:
    def __init__(self):
        self.player = {}
        self.cells = {}
        self.mode = None
        self.ui = None
        self.menus = 0
        self.version = None
        self.ended = False

    def apply(self, message):
        kind = message.get("msg")
        if kind == "client_path":
            self.version = message.get("version")
        elif kind == "version":
            text = message.get("text", "")
            prefix = "Dungeon Crawl Stone Soup "
            self.version = text[len(prefix):] if text.startswith(prefix) else None
        elif kind == "player":
            # Do not retain inventory, monster typedata, hidden statistics, etc.
            for key in ("hp", "hp_max", "mp", "mp_max", "turn", "pos", "place", "depth"):
                if key in message:
                    self.player[key] = copy.deepcopy(message[key])
        elif kind == "input_mode":
            self.mode = message["mode"]
        elif kind == "ui_state":
            self.ui = message["state"]
        elif kind == "menu":
            self.menus += 1
        elif kind == "close_menu":
            self.menus = max(0, self.menus - 1)
        elif kind == "init_menus":
            self.menus = len(message["menus"])
        elif kind == "exit_reason" and message.get("type") != "unknown":
            self.ended = True
        elif kind == "map":
            if message.get("clear"):
                self.cells.clear()
            x = y = None
            for delta in message.get("cells", []):
                if "x" in delta:
                    x, y = delta["x"], delta["y"]
                elif x is not None:
                    x += 1
                else:
                    raise ValueError("map delta without initial coordinates")
                cell = self.cells.setdefault((x, y), {"glyph": " ", "unseen": True, "monster": False})
                if "g" in delta:
                    cell["glyph"] = delta["g"]
                if "bg" in delta.get("t", {}):
                    cell["unseen"] = bool(tile_bits(delta["t"]["bg"]) & (UNSEEN | MM_UNSEEN))
                if "mon" in delta:
                    cell["monster"] = delta["mon"] is not None

    @property
    def ready(self):
        required = {"hp", "hp_max", "mp", "mp_max", "turn", "pos"}
        return (self.mode == COMMAND_MODE and self.ui == NORMAL_UI and self.menus == 0
                and not self.ended and required <= self.player.keys() and self.player["hp"] > 0)

    def snapshot(self, radius=2):
        if not self.ready:
            raise UnsafeInput(f"not gameplay input: mode={self.mode}, ui={self.ui}, menus={self.menus}, ended={self.ended}")
        pos = self.player["pos"]
        cells = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cell = self.cells.get((pos["x"] + dx, pos["y"] + dy))
                known = cell is not None and cell["glyph"] != " "
                visibility = "unknown" if not known else "remembered" if cell["unseen"] else "visible"
                cells.append({"dx": dx, "dy": dy, "visibility": visibility,
                              "glyph": cell["glyph"] if known else " ",
                              "monster": bool(known and visibility == "visible" and cell["monster"])})
        return {"player": copy.deepcopy(self.player), "cells": cells}


class LocalConnection:
    def __init__(self, game_socket, client_socket, process):
        self.process = process
        self.game_socket = str(game_socket)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(str(client_socket))
        self.decoder = MessageDecoder()
        self.state = ObservationState()
        self.pending = deque()

    def send(self, payload):
        self.socket.sendto(json.dumps(payload).encode(), self.game_socket)

    def attach(self):
        self.send({"msg": "attach", "primary": True})

    def receive_frame(self, deadline):
        """Consume through explicit flush marker, never guess readiness by silence."""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"DCSS exited with {self.process.returncode}")
            if not self.pending:
                data = self.read_data(min(0.25, max(0.001, deadline - time.monotonic())))
                if data is None:
                    continue
                self.pending.extend(self.decoder.feed(data))
            while self.pending:
                message = self.pending.popleft()
                self.state.apply(message)
                if message.get("msg") == "flush_messages":
                    return
        raise TimeoutError("DCSS observation deadline exceeded")

    def read_data(self, timeout):
        self.socket.settimeout(timeout)
        try:
            return self.socket.recv(128 * 1024)
        except socket.timeout:
            return None

    def act(self, action):
        if action not in ACTIONS:
            raise ValueError("unsupported action")
        if not self.state.ready:
            raise UnsafeInput("refusing gameplay key outside command mode")
        self.send({"msg": "key", "keycode": ord(ACTIONS[action])})

    def close(self):
        self.socket.close()


class PipeConnection(LocalConnection):
    """Same protocol through the disclosed transport-only 0.17.1 patch."""
    def __init__(self, read_fd, write_fd, process):
        self.process = process
        self.read_fd, self.write_fd = read_fd, write_fd
        self.decoder = MessageDecoder()
        self.state = ObservationState()
        self.pending = deque()

    def send(self, payload):
        data = json.dumps(payload).encode() + b"\n"
        while data:
            written = os.write(self.write_fd, data)
            data = data[written:]

    def read_data(self, timeout):
        readable, _, _ = select.select([self.read_fd], [], [], timeout)
        if not readable:
            return None
        data = os.read(self.read_fd, 128 * 1024)
        if not data:
            raise RuntimeError("DCSS protocol pipe closed")
        return data

    def close(self):
        os.close(self.read_fd)
        os.close(self.write_fd)
