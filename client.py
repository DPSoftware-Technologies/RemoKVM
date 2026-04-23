import array
import dearpygui.dearpygui as dpg
import pickle
import socket
import struct
import threading
import numpy as np
import cv2
import time
import brotli
import queue
import ssl

DPG_TO_HID = {
    dpg.mvKey_A: 0x04, dpg.mvKey_B: 0x05, dpg.mvKey_C: 0x06, dpg.mvKey_D: 0x07,
    dpg.mvKey_E: 0x08, dpg.mvKey_F: 0x09, dpg.mvKey_G: 0x0a, dpg.mvKey_H: 0x0b,
    dpg.mvKey_I: 0x0c, dpg.mvKey_J: 0x0d, dpg.mvKey_K: 0x0e, dpg.mvKey_L: 0x0f,
    dpg.mvKey_M: 0x10, dpg.mvKey_N: 0x11, dpg.mvKey_O: 0x12, dpg.mvKey_P: 0x13,
    dpg.mvKey_Q: 0x14, dpg.mvKey_R: 0x15, dpg.mvKey_S: 0x16, dpg.mvKey_T: 0x17,
    dpg.mvKey_U: 0x18, dpg.mvKey_V: 0x19, dpg.mvKey_W: 0x1a, dpg.mvKey_X: 0x1b,
    dpg.mvKey_Y: 0x1c, dpg.mvKey_Z: 0x1d,
    dpg.mvKey_1: 0x1e, dpg.mvKey_2: 0x1f, dpg.mvKey_3: 0x20, dpg.mvKey_4: 0x21,
    dpg.mvKey_5: 0x22, dpg.mvKey_6: 0x23, dpg.mvKey_7: 0x24, dpg.mvKey_8: 0x25,
    dpg.mvKey_9: 0x26, dpg.mvKey_0: 0x27,
    dpg.mvKey_Return: 0xB0, dpg.mvKey_Escape: 0xB1, dpg.mvKey_Back: 0xB2,
    dpg.mvKey_Tab: 0xB3, dpg.mvKey_Spacebar: 0x2c, dpg.mvKey_Capital: 0xC1,
    dpg.mvKey_Up: 0xDA, dpg.mvKey_Down: 0xD9, dpg.mvKey_Left: 0xD8,
    dpg.mvKey_Right: 0xD7, dpg.mvKey_Insert: 0xD1, dpg.mvKey_Delete: 0xD4,
    dpg.mvKey_Prior: 0xD3, dpg.mvKey_Next: 0xD6, dpg.mvKey_Home: 0xD2,
    dpg.mvKey_End: 0xD5,
    dpg.mvKey_LControl: 0x80, dpg.mvKey_LShift: 0x81, dpg.mvKey_LMenu: 0x82,
    dpg.mvKey_LWin: 0x83, dpg.mvKey_RControl: 0x84, dpg.mvKey_RShift: 0x85,
    dpg.mvKey_RMenu: 0x86, dpg.mvKey_RWin: 0x87,
    dpg.mvKey_Control: 0x80, dpg.mvKey_Shift: 0x81, dpg.mvKey_Alt: 0x82,
    dpg.mvKey_F1: 0xC2, dpg.mvKey_F2: 0xC3, dpg.mvKey_F3: 0xC4, dpg.mvKey_F4: 0xC5,
    dpg.mvKey_F5: 0xC6, dpg.mvKey_F6: 0xC7, dpg.mvKey_F7: 0xC8, dpg.mvKey_F8: 0xC9,
    dpg.mvKey_F9: 0xCA, dpg.mvKey_F10: 0xCB, dpg.mvKey_F11: 0xCC, dpg.mvKey_F12: 0xCD,
    dpg.mvKey_NumLock: 0xDB, dpg.mvKey_Divide: 0xDC, dpg.mvKey_Multiply: 0xDD,
    dpg.mvKey_Subtract: 0xDE, dpg.mvKey_Add: 0xDF,
    dpg.mvKey_NumPad1: 0xE1, dpg.mvKey_NumPad2: 0xE2, dpg.mvKey_NumPad3: 0xE3,
    dpg.mvKey_NumPad4: 0xE4, dpg.mvKey_NumPad5: 0xE5, dpg.mvKey_NumPad6: 0xE6,
    dpg.mvKey_NumPad7: 0xE7, dpg.mvKey_NumPad8: 0xE8, dpg.mvKey_NumPad9: 0xE9,
    dpg.mvKey_NumPad0: 0xEA, dpg.mvKey_Decimal: 0xEB,
    dpg.mvKey_Minus: 0x2d, dpg.mvKey_Plus: 0x2e, dpg.mvKey_Open_Brace: 0x2f,
    dpg.mvKey_Close_Brace: 0x30, dpg.mvKey_Backslash: 0x31, dpg.mvKey_Colon: 0x33,
    dpg.mvKey_Quote: 0x34, dpg.mvKey_Tilde: 0x35, dpg.mvKey_Comma: 0x36,
    dpg.mvKey_Period: 0x37, dpg.mvKey_Slash: 0x38,
    dpg.mvKey_Volume_Mute: 0x00E2, dpg.mvKey_Volume_Down: 0x00EA,
    dpg.mvKey_Volume_Up: 0x00E9, dpg.mvKey_Media_Play_Pause: 0x00CD,
    dpg.mvKey_Media_Stop: 0x00B7, dpg.mvKey_Media_Next_Track: 0x00B5,
    dpg.mvKey_Media_Prev_Track: 0x00B6, dpg.mvKey_Browser_Back: 0x0224,
    dpg.mvKey_Browser_Forward: 0x0225, dpg.mvKey_Browser_Refresh: 0x0227,
    dpg.mvKey_Browser_Stop: 0x0226, dpg.mvKey_Browser_Search: 0x0221,
    dpg.mvKey_Browser_Favorites: 0x022A, dpg.mvKey_Browser_Home: 0x0223,
    dpg.mvKey_Launch_Mail: 0x018A,
}

INITIAL_TEX_SIZE = (1280, 720)

class App:
    def __init__(self):
        self.connection_running = False
        self.socket = None
        self.reconnect_delay = 5
        self.server_res  = None   # (w, h) reported by decoded frames
        self.texture_res = None   # (w, h) of currently allocated DPG texture
        self.tex_tag = None
        self._layout_lock = threading.Lock()
        self._viewport_offset = (0, 0)
        self._viewport_draw_size = (0, 0)
        self.raw_buffer = queue.Queue(maxsize=3)  # compressed frames from TCP
        self.frame_buffer = queue.Queue(maxsize=2)   # decoded BGR numpy frames
        self.last_kvm_window_size = (0, 0)
        self.last_window_size = (0, 0)
        self.is_disconnected = True
        self.absolute_mouse = True
        self.last_cursor_pos = (0, 0)
        self.relative_mouse_sensitivity = 1.0

    def show_message(self, message):
        print(message)
        try:
            dpg.set_value("status_text", message)
        except Exception:
            pass

    def recv_exact(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.socket.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def tcp_receiver(self):
        while self.connection_running:
            try:
                raw_len = self.recv_exact(4)
                if not raw_len:
                    print("[TCP] connection closed")
                    break
                msg = struct.unpack('!I', raw_len)[0]
                data = self.recv_exact(msg)
                if not data:
                    break
                try:
                    self.raw_buffer.put_nowait(data)
                except queue.Full:
                    print("[TCP] raw_buffer full - dropping frame")
            except Exception as e:
                print(f"[TCP] {type(e).__name__}: {e}")
                self._do_reconnect()
                break

    def decompress_worker(self):
        print("[DECOMP] started")
        while self.connection_running:
            try:
                data = self.raw_buffer.get()

                # 12-byte metadata header, then compressed payload
                metadata = data[:13]
                metadatadata = struct.unpack('!III?', metadata)

                frame_bytes = data[13:]
                if metadatadata[3]: # is_compressed
                    try:
                        frame_bytes = brotli.decompress(frame_bytes)
                    except brotli.error as e:
                        print(f"[DECOMP] brotli error ({e}), using raw")

                enc = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if frame is None or frame.size == 0:
                    print("[DECOMP] imdecode failed")
                    continue

                # Keep only the freshest frame - drop stale ones
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_buffer.put_nowait(frame)

            except Exception as e:
                print(f"[DECOMP] {type(e).__name__}: {e}")

    def render_worker(self):
        print("[RENDER] worker started")
        while self.connection_running:
            try:
                frame = self.frame_buffer.get(timeout=0.05)
            except queue.Empty:
                continue

            frame_h, frame_w = frame.shape[:2]

            # Detect server resolution change
            if self.server_res != (frame_w, frame_h):
                self.server_res = (frame_w, frame_h)
                print(f"[RENDER] server res: {frame_w}×{frame_h}")
                self.show_message(f"Connected - stream: {frame_w}×{frame_h}")

            # Get current canvas size
            try:
                container_w = dpg.get_item_width("kvm_canvas")
                container_h = dpg.get_item_height("kvm_canvas")
            except Exception:
                continue

            if container_w <= 0 or container_h <= 0:
                continue

            # Letterbox computation
            draw_w, draw_h, off_x, off_y = self._compute_letterbox(container_w, container_h, frame_w, frame_h)

            with self._layout_lock:
                self._viewport_offset = (off_x, off_y)
                self._viewport_draw_size = (draw_w, draw_h)

            # Resize frame to draw size
            if (draw_w, draw_h) != (frame_w, frame_h):
                frame = cv2.resize(frame, (draw_w, draw_h),
                                   interpolation=cv2.INTER_LINEAR)

            # Reallocate texture if draw size changed
            if self.texture_res != (draw_w, draw_h):
                self._alloc_texture(draw_w, draw_h)

            # BGR → RGB, float32 normalised, flat
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tex_data = rgb.ravel().astype(np.float32) / 255.0

            try:
                dpg.set_value(self.tex_tag, tex_data)
                dpg.configure_item(
                    "kvm_draw_image",
                    pmin=(off_x, off_y),
                    pmax=(off_x + draw_w, off_y + draw_h),
                )
            except Exception as e:
                print(f"[RENDER] dpg error: {e}")

        print("[RENDER] worker stopped")

    def _alloc_texture(self, w, h):
        blank = array.array("f", [0.0] * (w * h * 3))
        new_tag = dpg.generate_uuid()

        with dpg.texture_registry():
            dpg.add_raw_texture(w, h, blank, format=dpg.mvFormat_Float_rgb, tag=new_tag)

        self.texture_res = (w, h)
        print(f"[TEXTURE] allocated {w}×{h}")

        if self.tex_tag is not None:
            old_tag = self.tex_tag
            self.tex_tag = new_tag
            try:
                dpg.configure_item("kvm_draw_image", texture_tag=self.tex_tag)
                if dpg.does_item_exist(old_tag):
                    dpg.delete_item(old_tag)
            except Exception:
                pass
        else:
            self.tex_tag = new_tag

    @staticmethod
    def _compute_letterbox(container_w, container_h, src_w, src_h):
        scale  = min(container_w / src_w, container_h / src_h)
        draw_w = int(src_w * scale)
        draw_h = int(src_h * scale)
        off_x  = (container_w - draw_w) // 2
        off_y  = (container_h - draw_h) // 2
        return draw_w, draw_h, off_x, off_y

    def connect_to_server(self, host, port, username, password):
        while self.connection_running:
            try:
                self.show_message(f"Connecting to {host}:{port}...")

                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket = ctx.wrap_socket(raw_sock, server_hostname=host)
                self.socket.connect((host, port))
                time.sleep(0.1)

                self.socket.sendall(pickle.dumps({"username": username, "password": password}))
                resp = self.socket.recv(1024)
                if not resp:
                    self.show_message("No response from server.")
                    self.socket.close()
                    break

                try:
                    result = pickle.loads(resp)
                    if result.get("success") == 1:
                        print("[AUTH] Login successful")
                except Exception as e:
                    print(f"[AUTH] decode error: {e}")

                # Spin up worker threads
                threading.Thread(target=self.tcp_receiver, daemon=True).start()
                threading.Thread(target=self.decompress_worker, daemon=True).start()
                threading.Thread(target=self.render_worker, daemon=True).start()

                dpg.show_item("disconnectbtn")
                self.show_message(f"Connected to {host}:{port}")
                #dpg.show_item("viewwindow")
                break

            except (socket.error, ConnectionRefusedError) as e:
                self.show_message(f"Failed to connect - retrying in {self.reconnect_delay}s...")
                dpg.show_item("disconnectbtn")
                print(f"[CONN] {e}")
                time.sleep(self.reconnect_delay)

    def _do_reconnect(self):
        if not self.is_disconnected:
            self.show_message("Reconnecting...")
            self.frame_buffer.queue.clear()
            try:
                self.socket.close()
            except Exception:
                pass
            self.connection_running = False
            self.connect(None, None)

    def convert_mouse_position(self, mouse_x, mouse_y):
        if self.server_res is None:
            return mouse_x, mouse_y

        with self._layout_lock:
            off_x,  off_y  = self._viewport_offset
            draw_w, draw_h = self._viewport_draw_size

        src_w, src_h = self.server_res
        if draw_w <= 0 or draw_h <= 0:
            return mouse_x, mouse_y

        rel_x = max(0, min(mouse_x - off_x, draw_w - 1))
        rel_y = max(0, min(mouse_y - off_y, draw_h - 1))
        return int(rel_x * src_w / draw_w), int(rel_y * src_h / draw_h)

    def send_action(self, action, **kwargs):
        try:
            payload = pickle.dumps({'action': action, 'data': kwargs})
            if self.socket:
                self.socket.sendall(struct.pack('!I', len(payload)) + payload)
        except (socket.error, BrokenPipeError):
            self._do_reconnect()

    def build_ui(self):
        with dpg.window(label="Remote", width=320, tag="remote_window"):
            with dpg.collapsing_header(label="Connection", default_open=True):
                dpg.add_input_text(label="IP", tag="ipinput", default_value="localhost")
                dpg.add_input_int(label="Port", tag="portinput",default_value=2222,  min_value=1, max_value=65535)
                dpg.add_spacer()
                dpg.add_input_text(label="Username", tag="usernameinput", default_value="admin")
                dpg.add_input_text(label="Password", tag="passwordinput", default_value="admin", password=True)
                dpg.add_spacer()
                dpg.add_button(label="Connect", callback=self.connect, tag="connectbtn")
                dpg.add_button(label="Disconnect", callback=self.disconnect, show=False, tag="disconnectbtn")
            with dpg.collapsing_header(label="Control"):
                dpg.add_combo(["Absolute", "Relative"], default_value="Absolute", tag="mousemodecombo", label="Mouse Mode", callback=lambda s, d: setattr(self, 'absolute_mouse', d == "Absolute"))
                dpg.add_input_float(label="Relative mouse sensitivity", tag="rel_sens_input", default_value=1.0, min_value=0.1, max_value=10.0, callback=lambda s, d: setattr(self, 'relative_mouse_sensitivity', d))
            with dpg.collapsing_header(label="I/O Control"):
                dpg.add_button(label="Power", tag="ctrlpwbtn")
                dpg.add_button(label="Reset", tag="ctrlrstbtn", callback=lambda: self.send_action('ioctrl', btn=1, state=2)) # press only

        # KVM window uses a drawlist - gives pixel-perfect control of image placement
        with dpg.window(label="KVM", tag="viewwindow", width=1280, height=720):
            with dpg.drawlist(width=1270, height=685, tag="kvm_canvas"):
                dpg.draw_image(self.tex_tag, pmin=(0, 0), pmax=(1270, 685), tag="kvm_draw_image")

        with dpg.window(label="About", tag="aboutwindow", show=False):
            dpg.add_text("RemoKVM Client")
            dpg.add_spacer()
            dpg.add_text("Copyright (C) 2026 DPSoftware Technologies. All rights reserved.")

    def build_menubar(self):
        with dpg.viewport_menu_bar(tag="menubar"):
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="About", callback=lambda: dpg.configure_item("aboutwindow", show=True))
            dpg.add_text("Ready", tag="status_text")

    def on_mouse_move(self, sender, mouse_pos):
        if dpg.is_item_hovered("kvm_canvas"):
            if self.absolute_mouse:
                pos = dpg.get_item_rect_min("kvm_canvas")
                ox, oy = self.convert_mouse_position(mouse_pos[0] - pos[0], mouse_pos[1] - pos[1])
                self.send_action('move_mouse', x=ox, y=oy)
            else:
                if self.last_cursor_pos == (0, 0):
                    self.last_cursor_pos = mouse_pos
                    return

                dx = (mouse_pos[0] - self.last_cursor_pos[0]) * self.relative_mouse_sensitivity
                dy = (mouse_pos[1] - self.last_cursor_pos[1]) * self.relative_mouse_sensitivity
                self.last_cursor_pos = mouse_pos
                self.send_action('move_relative_mouse', x=float("{:.3f}".format(dx)), y=float("{:.3f}".format(dy)))

    def on_key(self, sender, data):
        if dpg.is_item_hovered("kvm_canvas"):
            if isinstance(data, list):
                if data[1] == 0:
                    self.send_action('keyboard', state="down", key=DPG_TO_HID.get(data[0]))
            else:
                self.send_action('keyboard', state="up", key=DPG_TO_HID.get(data))

    def on_mouse_button(self, sender, data):
        if dpg.is_item_hovered("kvm_canvas"):
            if isinstance(data, list):
                if data[1] == 0:
                    # if 0 = 0, 1 = 2, 2 = 1 for some reason - remap to standard left=0, right=1, middle=2
                    if data[0] == 0:
                        btn = 0
                    elif data[0] == 1:
                        btn = 2
                    elif data[0] == 2:
                        btn = 1
                    else:
                        return
                    self.send_action('click_mouse', state="down", button=btn)
            else:
                if data == 0:
                    btn = 0
                elif data == 1:
                    btn = 2
                elif data == 2:
                    btn = 1
                else:
                    return
                self.send_action('click_mouse', state="up", button=btn)
        else:
            if isinstance(data, list):
                if data[1] == 0:
                    if dpg.is_item_hovered("ctrlpwbtn"):
                        self.send_action('ioctrl', btn=0, state=1)
            else:
                if dpg.is_item_hovered("ctrlpwbtn"):
                    self.send_action('ioctrl', btn=0, state=0)

    def on_mouse_wheel(self, sender, data):
        if dpg.is_item_hovered("kvm_canvas"):
            self.send_action('wheel_mouse', delta=data)

    def connect(self, _, __):
        ip = dpg.get_value("ipinput")
        port = dpg.get_value("portinput")
        username = dpg.get_value("usernameinput")
        password = dpg.get_value("passwordinput")

        for tag in ("ipinput", "portinput", "usernameinput", "passwordinput"):
            dpg.disable_item(tag)
        dpg.hide_item("connectbtn")
        self.connection_running = True
        self.is_disconnected = False
        threading.Thread(target=self.connect_to_server,args=(ip, port, username, password), daemon=True).start()

    def disconnect(self):
        self.connection_running = False
        self.is_disconnected = True
        #dpg.hide_item("viewwindow")
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
        self.show_message("Disconnected.")
        for tag in ("ipinput", "portinput", "usernameinput", "passwordinput"):
            dpg.enable_item(tag)
        dpg.show_item("connectbtn")
        dpg.hide_item("disconnectbtn")

    def init(self):
        dpg.create_context()
        dpg.create_viewport(title='RemoKVM', width=1600, height=900)
        dpg.setup_dearpygui()

        # Allocate initial placeholder texture
        self._alloc_texture(*INITIAL_TEX_SIZE)

        with dpg.handler_registry():
            dpg.add_mouse_move_handler(callback=self.on_mouse_move)
            dpg.add_key_down_handler(callback=self.on_key)
            dpg.add_key_release_handler(callback=self.on_key)
            dpg.add_mouse_down_handler(callback=self.on_mouse_button)
            dpg.add_mouse_release_handler(callback=self.on_mouse_button)
            dpg.add_mouse_wheel_handler(callback=self.on_mouse_wheel)

        self.build_menubar()
        self.build_ui()

        dpg.configure_app(docking=True, docking_space=True)
        dpg.configure_app(init_file="workspace.ini")

        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            self._sync_layout()
            dpg.render_dearpygui_frame()

        self.exit()

    def _sync_layout(self):
        window_w = dpg.get_viewport_width()
        window_h = dpg.get_viewport_height()

        kvm_w = dpg.get_item_width("viewwindow")
        kvm_h = dpg.get_item_height("viewwindow")

        if self.last_kvm_window_size != (kvm_w, kvm_h):
            canvas_w = max(1, kvm_w - 16)
            canvas_h = max(1, kvm_h - 38)
            dpg.configure_item("kvm_canvas", width=canvas_w, height=canvas_h)
            self.last_kvm_window_size = (kvm_w, kvm_h)

        if self.last_window_size != (window_w, window_h):
            dpg.set_item_pos("status_text", [window_w - 320, 0])
            self.last_window_size = (window_w, window_h)

    def exit(self):
        self.disconnect()
        dpg.destroy_context()


app = App()
app.init()