import socket
import struct
import pickle

HEADER_SIZE = 8  # длина заголовка с размером сообщения


def send_message(sock: socket.socket, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!Q", len(data))
    sock.sendall(header + data)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    received = 0
    while received < n:
        chunk = sock.recv(n - received)
        if not chunk:
            raise ConnectionError("Соединение закрыто во время чтения")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket):
    header = recv_exact(sock, HEADER_SIZE)
    (length,) = struct.unpack("!Q", header)
    payload = recv_exact(sock, length)
    return pickle.loads(payload)