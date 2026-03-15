import os
import sys
import socket
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext


class SLAUApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распределённое решение СЛАУ")
        self.root.geometry("1050x740")
        self.root.minsize(950, 650)

        self.python_exec = sys.executable
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        self.worker1_proc = None
        self.worker2_proc = None

        self.create_widgets()
        self.log(f"[INFO] Python: {self.python_exec}")
        self.log(f"[INFO] Проект: {self.project_dir}")
        self.log("[INFO] Для запуска master.py и benchmark.py сначала запустите оба worker-а (5001 и 5002).")
        self.log("")

        self.update_run_buttons()

    def create_widgets(self):
        title = ttk.Label(
            self.root,
            text="Программный комплекс для распределённого решения СЛАУ",
            font=("Segoe UI", 15, "bold")
        )
        title.pack(pady=10)

        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Размерность n:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.n_var = tk.StringVar(value="100")
        ttk.Entry(top_frame, textvariable=self.n_var, width=12).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(top_frame, text="Benchmark sizes:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.sizes_var = tk.StringVar(value="100 300 1000 2000 5000")
        ttk.Entry(top_frame, textvariable=self.sizes_var, width=32).grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(top_frame, text="dist-limit:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.dist_limit_var = tk.StringVar(value="300")
        ttk.Entry(top_frame, textvariable=self.dist_limit_var, width=10).grid(row=0, column=5, padx=5, pady=5, sticky="w")

        btn_frame1 = ttk.Frame(self.root, padding=10)
        btn_frame1.pack(fill="x")

        self.btn_generate = ttk.Button(btn_frame1, text="1. Сгенерировать данные", command=self.generate_data)
        self.btn_generate.grid(row=0, column=0, padx=5, pady=5)

        self.btn_worker1 = ttk.Button(btn_frame1, text="2. Запустить Worker 5001", command=self.start_worker1)
        self.btn_worker1.grid(row=0, column=1, padx=5, pady=5)

        self.btn_worker2 = ttk.Button(btn_frame1, text="3. Запустить Worker 5002", command=self.start_worker2)
        self.btn_worker2.grid(row=0, column=2, padx=5, pady=5)

        self.btn_stop_workers = ttk.Button(btn_frame1, text="Остановить Workers", command=self.stop_workers)
        self.btn_stop_workers.grid(row=0, column=3, padx=5, pady=5)

        btn_frame2 = ttk.Frame(self.root, padding=10)
        btn_frame2.pack(fill="x")

        self.btn_master = ttk.Button(btn_frame2, text="4. Запустить master.py", command=self.run_master)
        self.btn_master.grid(row=0, column=0, padx=5, pady=5)

        self.btn_benchmark = ttk.Button(btn_frame2, text="5. Запустить benchmark.py", command=self.run_benchmark)
        self.btn_benchmark.grid(row=0, column=1, padx=5, pady=5)

        self.btn_test_gauss = ttk.Button(btn_frame2, text="Тест Gauss", command=self.run_test_gauss)
        self.btn_test_gauss.grid(row=0, column=2, padx=5, pady=5)

        self.btn_test_orth = ttk.Button(btn_frame2, text="Тест Orth", command=self.run_test_orth)
        self.btn_test_orth.grid(row=0, column=3, padx=5, pady=5)

        self.btn_load_test = ttk.Button(btn_frame2, text="Load Test", command=self.run_load_test)
        self.btn_load_test.grid(row=0, column=4, padx=5, pady=5)

        info_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        info_frame.pack(fill="x")

        info_text = (
            "Примечание:\n"
            "• master.py работает с файлами data/Matrix.txt и data/Vector.txt.\n"
            "• benchmark.py запускается только при активных worker-ах.\n"
            "• Для распределённого режима обязательно запустите Worker 5001 и Worker 5002."
        )
        ttk.Label(info_frame, text=info_text, justify="left").pack(anchor="w")

        log_frame = ttk.Frame(self.root, padding=10)
        log_frame.pack(fill="both", expand=True)

        ttk.Label(log_frame, text="Лог выполнения:", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.log_text.pack(fill="both", expand=True, pady=5)

        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(fill="x")

        ttk.Button(bottom_frame, text="Очистить лог", command=self.clear_log).pack(side="left", padx=5)
        ttk.Button(bottom_frame, text="Выход", command=self.on_close).pack(side="right", padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def log(self, text):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def file_exists(self, filename):
        return os.path.exists(os.path.join(self.project_dir, filename))

    def get_subprocess_encoding(self):
        if os.name == "nt":
            return "cp1251"
        return "utf-8"

    def is_port_open(self, host, port, timeout=0.5):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    def workers_ready(self):
        w1 = self.is_port_open("127.0.0.1", 5001)
        w2 = self.is_port_open("127.0.0.1", 5002)
        return w1 and w2

    def update_run_buttons(self):
        if self.workers_ready():
            self.btn_master.config(state="normal")
            self.btn_benchmark.config(state="normal")
        else:
            self.btn_master.config(state="disabled")
            self.btn_benchmark.config(state="disabled")

    def run_command_in_thread(self, cmd, title=None):
        def target():
            try:
                if title:
                    self.log("=" * 80)
                    self.log(title)
                    self.log("=" * 80)

                self.log(f"[CMD] {' '.join(cmd)}")
                self.log("")

                process = subprocess.Popen(
                    cmd,
                    cwd=self.project_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding=self.get_subprocess_encoding(),
                    errors="replace"
                )

                for line in process.stdout:
                    self.log(line.rstrip())

                process.wait()
                self.log("")
                self.log(f"[ЗАВЕРШЕНО] Код возврата: {process.returncode}")
                self.log("")

            except Exception as e:
                self.log(f"[ОШИБКА] {e}")
                self.log("")

        threading.Thread(target=target, daemon=True).start()

    def generate_data(self):
        if not self.file_exists("data_generator.py"):
            messagebox.showerror("Ошибка", "Файл data_generator.py не найден.")
            return

        n = self.n_var.get().strip()
        if not n.isdigit():
            messagebox.showerror("Ошибка", "Размерность n должна быть целым положительным числом.")
            return

        cmd = [self.python_exec, "data_generator.py", "--n", n]
        self.run_command_in_thread(cmd, title=f"ГЕНЕРАЦИЯ ДАННЫХ (n={n})")

    def start_worker1(self):
        if not self.file_exists("worker.py"):
            messagebox.showerror("Ошибка", "Файл worker.py не найден.")
            return

        if self.worker1_proc and self.worker1_proc.poll() is None:
            self.log("[INFO] Worker 5001 уже запущен.")
            self.update_run_buttons()
            return

        try:
            self.worker1_proc = subprocess.Popen(
                [self.python_exec, "worker.py", "--port", "5001"],
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding=self.get_subprocess_encoding(),
                errors="replace"
            )
            self.log("[INFO] Worker 5001 запущен.")
            threading.Thread(target=self._read_worker_output, args=(self.worker1_proc, "WORKER-5001"), daemon=True).start()
            self.root.after(700, self.update_run_buttons)
        except Exception as e:
            self.log(f"[ОШИБКА] Не удалось запустить Worker 5001: {e}")

    def start_worker2(self):
        if not self.file_exists("worker.py"):
            messagebox.showerror("Ошибка", "Файл worker.py не найден.")
            return

        if self.worker2_proc and self.worker2_proc.poll() is None:
            self.log("[INFO] Worker 5002 уже запущен.")
            self.update_run_buttons()
            return

        try:
            self.worker2_proc = subprocess.Popen(
                [self.python_exec, "worker.py", "--port", "5002"],
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding=self.get_subprocess_encoding(),
                errors="replace"
            )
            self.log("[INFO] Worker 5002 запущен.")
            threading.Thread(target=self._read_worker_output, args=(self.worker2_proc, "WORKER-5002"), daemon=True).start()
            self.root.after(700, self.update_run_buttons)
        except Exception as e:
            self.log(f"[ОШИБКА] Не удалось запустить Worker 5002: {e}")

    def _read_worker_output(self, process, tag):
        try:
            for line in process.stdout:
                self.log(f"[{tag}] {line.rstrip()}")
        except Exception as e:
            self.log(f"[ОШИБКА {tag}] {e}")
        finally:
            self.root.after(300, self.update_run_buttons)

    def stop_workers(self):
        stopped = False

        if self.worker1_proc and self.worker1_proc.poll() is None:
            self.worker1_proc.terminate()
            self.log("[INFO] Worker 5001 остановлен.")
            stopped = True

        if self.worker2_proc and self.worker2_proc.poll() is None:
            self.worker2_proc.terminate()
            self.log("[INFO] Worker 5002 остановлен.")
            stopped = True

        if not stopped:
            self.log("[INFO] Нет активных worker-процессов.")

        self.root.after(500, self.update_run_buttons)

    def run_master(self):
        if not self.file_exists("master.py"):
            messagebox.showerror("Ошибка", "Файл master.py не найден.")
            return

        if not self.workers_ready():
            messagebox.showwarning(
                "Worker-ы не запущены",
                "Для запуска master.py необходимо сначала запустить Worker 5001 и Worker 5002."
            )
            self.update_run_buttons()
            return

        matrix_path = os.path.join(self.project_dir, "data", "Matrix.txt")
        vector_path = os.path.join(self.project_dir, "data", "Vector.txt")

        if not os.path.exists(matrix_path) or not os.path.exists(vector_path):
            messagebox.showerror(
                "Ошибка",
                "Файлы data/Matrix.txt и/или data/Vector.txt не найдены.\n"
                "Сначала выполните генерацию данных."
            )
            return

        cmd = [self.python_exec, "master.py"]
        self.run_command_in_thread(cmd, title="ЗАПУСК MASTER.PY")

    def run_benchmark(self):
        if not self.file_exists("benchmark.py"):
            messagebox.showerror("Ошибка", "Файл benchmark.py не найден.")
            return

        if not self.workers_ready():
            messagebox.showwarning(
                "Worker-ы не запущены",
                "Для запуска benchmark.py необходимо сначала запустить Worker 5001 и Worker 5002."
            )
            self.update_run_buttons()
            return

        sizes = self.sizes_var.get().strip()
        dist_limit = self.dist_limit_var.get().strip()

        if not dist_limit.isdigit():
            messagebox.showerror("Ошибка", "dist-limit должен быть целым числом.")
            return

        size_parts = sizes.split()
        if not size_parts or not all(part.isdigit() for part in size_parts):
            messagebox.showerror("Ошибка", "Benchmark sizes должны содержать целые числа через пробел.")
            return

        cmd = [self.python_exec, "benchmark.py", "--sizes", *size_parts, "--dist-limit", dist_limit]
        self.run_command_in_thread(cmd, title="ЗАПУСК BENCHMARK.PY")

    def run_test_gauss(self):
        if not self.file_exists("tests/test_gauss.py"):
            messagebox.showwarning("Файл не найден", "Файл test_gauss.py отсутствует.")
            self.log("[WARN] test_gauss.py не найден.")
            return

        cmd = [self.python_exec, "test_gauss.py"]
        self.run_command_in_thread(cmd, title="МОДУЛЬНЫЙ ТЕСТ: TEST_GAUSS.PY")

    def run_test_orth(self):
        if not self.file_exists("tests/test_orth.py"):
            messagebox.showwarning("Файл не найден", "Файл test_orth.py отсутствует.")
            self.log("[WARN] test_orth.py не найден.")
            return

        cmd = [self.python_exec, "test_orth.py"]
        self.run_command_in_thread(cmd, title="МОДУЛЬНЫЙ ТЕСТ: TEST_ORTH.PY")

    def run_load_test(self):
        if not self.file_exists("tests/load_test.py"):
            messagebox.showwarning("Файл не найден", "Файл load_test.py отсутствует.")
            self.log("[WARN] load_test.py не найден.")
            return

        cmd = [self.python_exec, "load_test.py"]
        self.run_command_in_thread(cmd, title="НАГРУЗОЧНЫЙ ТЕСТ: LOAD_TEST.PY")

    def on_close(self):
        self.stop_workers()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    SLAUApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()