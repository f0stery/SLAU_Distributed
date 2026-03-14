import argparse
import socket
import pickle
import numpy as np


def recv_all(conn, size):
    chunks = []
    received = 0
    while received < size:
        chunk = conn.recv(min(4096, size - received))
        if not chunk:
            break
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def handle_request(payload):
    cmd = payload["cmd"]

    if cmd == "get_column_and_norm":
        A_chunk = payload["A_chunk"]
        k = payload["k"]
        col_k = A_chunk[:, k].copy()
        norm_sq = float(np.dot(col_k, col_k))
        return {
            "col_k": col_k,
            "norm_sq": norm_sq,
        }

    elif cmd == "dot_with_q":
        A_chunk = payload["A_chunk"]
        q_chunk = payload["q_chunk"]
        j = payload["j"]
        dot = float(np.dot(q_chunk, A_chunk[:, j]))
        return {
            "dot": dot,
        }

    elif cmd == "update_column":
        A_chunk = payload["A_chunk"]
        q_chunk = payload["q_chunk"]
        j = payload["j"]
        rkj = payload["rkj"]

        A_chunk = A_chunk.copy()
        A_chunk[:, j] = A_chunk[:, j] - rkj * q_chunk

        return {
            "A_chunk": A_chunk,
        }

    else:
        return {
            "error": f"Неизвестная команда: {cmd}"
        }


def main():
    parser = argparse.ArgumentParser(description="Worker-узел для распределённого MGS")
    parser.add_argument("--host", default="0.0.0.0", help="Адрес для прослушивания")
    parser.add_argument("--port", type=int, required=True, help="Порт для прослушивания")
    args = parser.parse_args()

    host = args.host
    port = args.port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((host, port))
        server.listen()

        print(f"[WORKER] Запущен на {host}:{port}")

        while True:
            conn, addr = server.accept()
            with conn:
                size_bytes = recv_all(conn, 8)
                if len(size_bytes) < 8:
                    continue

                size = int.from_bytes(size_bytes, "big")
                data = recv_all(conn, size)
                if len(data) < size:
                    continue

                payload = pickle.loads(data)
                result = handle_request(payload)

                response = pickle.dumps(result)
                conn.sendall(len(response).to_bytes(8, "big"))
                conn.sendall(response)


if __name__ == "__main__":
    main()