import argparse
import pickle
import socket
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# ---------------------------
# Framed TCP helpers
# ---------------------------

def recv_exact(conn: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Соединение закрыто во время чтения данных")
        data.extend(chunk)
    return bytes(data)


def recv_message(conn: socket.socket):
    header = recv_exact(conn, 8)
    (length,) = struct.unpack("!Q", header)
    payload = recv_exact(conn, length)
    return pickle.loads(payload)


def send_message(conn: socket.socket, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!Q", len(payload))
    conn.sendall(header + payload)


# ---------------------------
# Worker state
# ---------------------------

STATE = {
    "A_chunk": None,       # shape (m_i, n)
    "b_chunk": None,       # shape (m_i,)
    "row_start": None,
    "row_end": None,
    "m_local": None,
    "n": None,
    "dtype": None,
}


def handle_request(req: dict) -> dict:
    cmd = req.get("cmd")

    if cmd == "ping":
        return {"status": "ok"}

    if cmd == "init_data":
        A_chunk = np.asarray(req["A_chunk"], dtype=np.float64)
        b_chunk = np.asarray(req["b_chunk"], dtype=np.float64)
        row_start = int(req["row_start"])
        row_end = int(req["row_end"])

        if A_chunk.ndim != 2:
            return {"error": "A_chunk должен быть двумерным"}
        if b_chunk.ndim != 1:
            return {"error": "b_chunk должен быть одномерным"}
        if A_chunk.shape[0] != b_chunk.shape[0]:
            return {"error": "Число строк A_chunk и длина b_chunk должны совпадать"}

        STATE["A_chunk"] = A_chunk.copy()
        STATE["b_chunk"] = b_chunk.copy()
        STATE["row_start"] = row_start
        STATE["row_end"] = row_end
        STATE["m_local"] = A_chunk.shape[0]
        STATE["n"] = A_chunk.shape[1]
        STATE["dtype"] = str(A_chunk.dtype)

        return {
            "status": "ok",
            "m_local": STATE["m_local"],
            "n": STATE["n"],
            "row_start": row_start,
            "row_end": row_end,
        }

    if STATE["A_chunk"] is None:
        return {"error": "Данные не инициализированы. Сначала вызовите init_data."}

    A_chunk = STATE["A_chunk"]
    b_chunk = STATE["b_chunk"]
    n = STATE["n"]

    if cmd == "get_col_norm":
        k = int(req["k"])
        if not (0 <= k < n):
            return {"error": f"Некорректный k={k}"}

        col_local = A_chunk[:, k].copy()
        norm_sq_local = float(np.dot(col_local, col_local))

        return {
            "col_local": col_local,
            "norm_sq_local": norm_sq_local,
        }

    if cmd == "compute_dots":
        k = int(req["k"])
        q_local = np.asarray(req["q_local"], dtype=np.float64)

        if q_local.shape[0] != A_chunk.shape[0]:
            return {"error": "Размер q_local не совпадает с числом локальных строк"}

        if k + 1 >= n:
            return {"dots_local": np.empty((0,), dtype=np.float64)}

        # Векторизованно: q_local @ A_chunk[:, k+1:]
        dots_local = q_local @ A_chunk[:, (k + 1):]

        return {
            "dots_local": np.asarray(dots_local, dtype=np.float64),
        }

    if cmd == "update_tail":
        k = int(req["k"])
        q_local = np.asarray(req["q_local"], dtype=np.float64)
        r_tail = np.asarray(req["r_tail"], dtype=np.float64)

        if q_local.shape[0] != A_chunk.shape[0]:
            return {"error": "Размер q_local не совпадает с числом локальных строк"}

        tail_cols = n - (k + 1)
        if tail_cols <= 0:
            return {"status": "ok"}

        if r_tail.shape[0] != tail_cols:
            return {"error": f"Размер r_tail должен быть {tail_cols}, получено {r_tail.shape[0]}"}

        # Векторизованное обновление:
        # A_chunk[:, k+1:] -= np.outer(q_local, r_tail)
        A_chunk[:, (k + 1):] -= np.outer(q_local, r_tail)

        return {"status": "ok"}

    if cmd == "compute_yk":
        q_local = np.asarray(req["q_local"], dtype=np.float64)

        if q_local.shape[0] != A_chunk.shape[0]:
            return {"error": "Размер q_local не совпадает с числом локальных строк"}

        yk_local = float(np.dot(q_local, b_chunk))
        return {"yk_local": yk_local}

    if cmd == "shutdown":
        return {"status": "bye"}

    return {"error": f"Неизвестная команда: {cmd}"}


def handle_client(conn: socket.socket, addr):
    try:
        req = recv_message(conn)
        resp = handle_request(req)
        send_message(conn, resp)
    except Exception as e:
        try:
            send_message(conn, {"error": str(e)})
        except Exception:
            pass
    finally:
        conn.close()


def serve(host: str, port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        print(f"[WORKER] Запущен на {host}:{port}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                conn, addr = server.accept()
                pool.submit(handle_client, conn, addr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCP worker для распределённого MGS")
    parser.add_argument("--host", default="0.0.0.0", help="Хост для bind")
    parser.add_argument("--port", type=int, required=True, help="Порт для worker")
    args = parser.parse_args()

    serve(args.host, args.port)