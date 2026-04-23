import socket
import threading
import brotli
import numpy as np
import cv2
import configparser
import struct
import queue
import pickle
import ssl
import time
from KVMcontroller import PicoController

# Read configuration
config = configparser.ConfigParser()
config.read('serverconfig.ini')

# stream config
jpegquality = int(config["stream"]["quality"])
formatcodec = config["stream"]["format"]
resX = int(config["stream"]["x"])
resY = int(config["stream"]["y"])
fps = int(config["stream"]["fps"])
compression = int(config["stream"]["compression"])

# audio config
aenable = bool(int(config["audio"]["enable"]))
abitrate = int(config["audio"]["bitrate"])

# server config
HOST = config["server"]["ip"]
PORT = int(config["server"]["port"])

# video in config
AUTOSUSPEND = bool(int(config["videoin"]["autosuspend"]))
video_device_path = config["videoin"]["VideoDeviceIn"]
video_usb_id = config["videoin"]["USBUVCid"]
device_not_ready_image = config["videoin"]["devicenotreadyimage"]

# KVM controller config
controller_serial_port = config["KVM"]["controlSerialPort"]
controller_baudrate = int(config["KVM"]["baudrate"])

# Rate limiting config
MAX_COMMAND_QUEUE = int(config["KVM"]["maxCommandQueueSize"])  # Max queued commands before dropping
COMMAND_RATE_LIMIT = int(config["KVM"]["commandRateLimit"])  # Max commands per second

userdb = [
    {"username": "admin", "password": "admin", "permissions": ["keyboard", "mouse", "view", "powerio", "agent"]},
    {"username": "viewer", "password": "viewer", "permissions": ["view"]}
]

screensize = ()

cap = None
iscapready = False

deviceUnreadyImage = cv2.imread(device_not_ready_image)

KVMCtrl = PicoController(port=controller_serial_port, baud=controller_baudrate)
KVMCtrl.connect()

# Command queue for rate limiting and buffering
command_queue = queue.Queue(maxsize=MAX_COMMAND_QUEUE)
command_queue_lock = threading.Lock()
last_command_time = 0.0
commands_processed = 0

def set_usb_autosuspend(usb_id, enable=True):
    # path usually looks like /sys/bus/usb/devices/1-1/power/control
    path = f"/sys/bus/usb/devices/{usb_id}/power/control"
    mode = "auto" if enable else "on"

    try:
        with open(path, 'w') as f:
            f.write(mode)
    except PermissionError:
        print("Error: Need sudo privileges to change USB power settings.")

def start_video_stream():
    global cap
    if AUTOSUSPEND:
        print(f"Waking up USB device: {video_usb_id}")
        set_usb_autosuspend(video_usb_id, enable=False)

    # Open the capture device only when needed
    cap = cv2.VideoCapture(video_device_path, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, fps)

def stop_video_stream():
    global cap
    if cap:
        cap.release()
        cap = None

    if AUTOSUSPEND:
        print(f"Suspending USB device: {video_usb_id}")
        set_usb_autosuspend(video_usb_id, enable=True)

def imagenc(image, quality=90):
    if formatcodec == "webp":
        retval, buffer = cv2.imencode('.webp', image, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
    elif formatcodec == "jpeg":
        retval, buffer = cv2.imencode('.jpeg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    elif formatcodec == "avif":
        retval, buffer = cv2.imencode('.avif', image, [int(cv2.IMWRITE_AVIF_QUALITY), quality])

    else:
        raise TypeError(f"{formatcodec} is not supported")

    if not retval:
        raise ValueError("image encoding failed.")

    return np.array(buffer).tobytes()

def translate_coordinates(x, y, resized_width, resized_height):
    translated_x = int(x * screensize[0] / resized_width)
    translated_y = int(y * screensize[1] / resized_height)
    return translated_x, translated_y

def convert_quality(quality):
    brotli_quality = int(quality / 100 * 11)
    lgwin = int(10 + (quality / 100 * (24 - 10)))

    return brotli_quality, lgwin

client_sockets = []
buffer = queue.Queue(maxsize=2)
first = True
running = False

def capture():
    while running:
        # Capture the screen
        if cap is None or not cap.isOpened():
            screen_image = deviceUnreadyImage
        else:
            ret, screen_image = cap.read() # Replaced capture_screen() with cap.read()
            if not ret:
                continue

        # Resize the image
        stretch_near = cv2.resize(screen_image, (resX, resY), interpolation=cv2.INTER_NEAREST)

        # Encode the image
        encoded = imagenc(stretch_near, jpegquality)

        if compression > 0:
            bquality, lgwin = convert_quality(compression)

            compressed = brotli.compress(encoded, quality=bquality, lgwin=lgwin)
        else:
            compressed = encoded

        data_length = struct.pack('!III?', len(compressed), resX, resY, compression > 0)
        data2send = data_length + compressed

        buffer.put(data2send)

def send_frame(data, client):
    length = struct.pack('!I', len(data))
    client.sendall(length + data)

def handle_client():
    global running, first
    try:
        while running:
            data2send = buffer.get()

            dead = []
            for client in client_sockets:
                try:
                    send_frame(data2send, client)
                except Exception as e:
                    print(f"send error: {e}")
                    client.close()
                    dead.append(client)

            for dc in dead:
                client_sockets.remove(dc)

            if not client_sockets:
                running = False
                first = True
                stop_video_stream()
                print("No clients connected. Server is standby")
                break

    except Exception as e:
        print(f"Error in handle_client: {e}")

def enqueue_command(command, user):
    """Enqueue command with rate limiting. Returns False if queue is full (command dropped)."""
    global last_command_time, commands_processed
    
    try:
        command_queue.put_nowait((command, user))
        return True
    except queue.Full:
        print(f"[RATE LIMIT] Command queue full - dropping command: {command.get('action')}")
        return False

def command_processor():
    """Process queued commands with rate limiting to Pico controller."""
    global last_command_time, commands_processed
    
    min_interval = 1.0 / COMMAND_RATE_LIMIT
    
    while running:
        try:
            command, user = command_queue.get(timeout=0.1)
            
            # Rate limit: enforce minimum interval between commands
            elapsed = time.time() - last_command_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            handle_command_sync(command, user)
            last_command_time = time.time()
            commands_processed += 1
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error processing command: {e}")

def handle_command_sync(command, user):
    """Synchronously execute a command on the KVM controller."""
    action = command["action"]
    data = command["data"]

    # get permissions of the user
    permissions = next((u["permissions"] for u in userdb if u["username"] == user), [])

    try:
        if action == "move_mouse" and "mouse" in permissions:
            logicalX = (data["x"] * 32767) / resX
            logicalY = (data["y"] * 32767) / resY
            KVMCtrl.mouse_move_abs(int(logicalX), int(logicalY))
        elif action == "move_relative_mouse" and "mouse" in permissions:
            dx = int(data["x"])
            dy = int(data["y"])
            KVMCtrl.mouse_move_rel(dx, dy)
        elif action == "click_mouse" and "mouse" in permissions:
            if data["state"] == "down":
                KVMCtrl.mouse_hold(data["button"])
            elif data["state"] == "up":
                KVMCtrl.mouse_release(data["button"])
        elif action == "wheel_mouse" and "mouse" in permissions:
            KVMCtrl.mouse_scroll(data["delta"])
        elif action == "keyboard" and "keyboard" in permissions:
            if data["state"] == "down":
                KVMCtrl.key_hold(data["key"])
            elif data["state"] == "up":
                KVMCtrl.key_release(data["key"])
        elif action == "ioctrl" and "powerio" in permissions:
            if data["state"] == 1 and data["btn"] == 0:
                KVMCtrl.power_hold()
            elif data["state"] == 0 and data["btn"] == 0:
                KVMCtrl.power_release()
            elif data["state"] == 2 and data["btn"] == 1:
                KVMCtrl.reset_press()
    except Exception as e:
        print(f"Error executing command {action}: {e}")

def handle_commands(command, user):
    """Non-blocking command enqueuing (returns immediately)."""
    enqueue_command(command, user)

def handle_client_commands(client_socket, user):
    try:
        while True:
            try:
                # Receive the length of the data
                data_length = receive_exact(client_socket, 4)
                if not data_length:
                    break

                commandmetadata = struct.unpack('!I', data_length)
                command_data = receive_exact(client_socket, commandmetadata[0])
                command = pickle.loads(command_data)

                if command:
                    handle_commands(command, user)

            except socket.error:
                break
    except Exception as e:
        raise e

def receive_exact(socket, n):
    """Helper function to receive exactly n bytes."""
    data = b''
    while len(data) < n:
        packet = socket.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain(certfile="server.crt", keyfile="server.key")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen()

print(f"Server started on {HOST}:{PORT}")
try:
    while True:
        conn, addr = s.accept()
        print(f'{addr} is connected')
        # perform authentication
        secure_conn = context.wrap_socket(conn, server_side=True)

        auth_data = secure_conn.recv(1024)
        auth_data = pickle.loads(auth_data)

        # check if the user is exist and the password is correct
        user = next((u for u in userdb if u["username"] == auth_data["username"] and u["password"] == auth_data["password"]), None)
        if user:
            print(f"User {auth_data['username']} authenticated successfully.")
            secure_conn.sendall(pickle.dumps({"success": 1}))
        else:
            print(f"User {auth_data['username']} failed to authenticate.")
            secure_conn.sendall(pickle.dumps({"success": 0}))
            secure_conn.close()
            continue

        client_sockets.append(secure_conn)

        if first:
            running = True
            # Start the capture thread
            capture_thread = threading.Thread(target=capture, daemon=True)
            handle_client_thread = threading.Thread(target=handle_client, daemon=True)
            command_worker_thread = threading.Thread(target=command_processor, daemon=True)
            capture_thread.start()
            handle_client_thread.start()
            command_worker_thread.start()
            start_video_stream()

            first = False

        command_thread = threading.Thread(target=handle_client_commands, args=(secure_conn, auth_data['username']))
        command_thread.start()
except KeyboardInterrupt:
    print("Shutting down server...")
finally:
    stop_video_stream()
    for client in client_sockets:
        client.close()
    s.close()