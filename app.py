import os
import sys
import socket
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import queue
import signal


class SLAUApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распределённое решение СЛАУ")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)

        self.python_exec = sys.executable
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        self.worker1_proc = None
        self.worker2_proc = None
        self.current_process = None
        self.is_running = False

        self.log_queue = queue.Queue()
        self.running_processes = []

        self.create_widgets()
        self.process_log_queue()

        self.log("[INFO] Программа запущена. Готов к работе.\n")
        self.update_run_buttons()

    # ---------------------- GUI ----------------------
    def create_widgets(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_paned, width=450)
        main_paned.add(left_frame, weight=1)

        ttk.Label(left_frame, text="Распределённое решение СЛАУ",
                  font=("Segoe UI", 12, "bold")).pack(pady=10)

        # Параметры
        params_frame = ttk.LabelFrame(left_frame, text="Параметры", padding=10)
        params_frame.pack(fill="x", padx=10, pady=5)
        n_frame = ttk.Frame(params_frame)
        n_frame.pack(fill="x", pady=2)
        ttk.Label(n_frame, text="Размерность n:", width=15).pack(side="left")
        self.n_var = tk.StringVar(value="100")
        ttk.Entry(n_frame, textvariable=self.n_var, width=15).pack(side="left", padx=5)

        sizes_frame = ttk.Frame(params_frame)
        sizes_frame.pack(fill="x", pady=2)
        ttk.Label(sizes_frame, text="Benchmark sizes:", width=15).pack(side="left")
        self.sizes_var = tk.StringVar(value="10 30 50 100 300 1000 2000 5000")
        ttk.Entry(sizes_frame, textvariable=self.sizes_var, width=30).pack(side="left", padx=5)

        dist_frame = ttk.Frame(params_frame)
        dist_frame.pack(fill="x", pady=2)
        ttk.Label(dist_frame, text="dist-limit:", width=15).pack(side="left")
        self.dist_limit_var = tk.StringVar(value="300")
        ttk.Entry(dist_frame, textvariable=self.dist_limit_var, width=15).pack(side="left", padx=5)

        # Статус
        status_frame = ttk.LabelFrame(left_frame, text="Статус", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        self.status_label = ttk.Label(status_frame, text="⚫ Готов к работе", foreground="green",
                                      font=("Segoe UI", 10, "bold"))
        self.status_label.pack(anchor="w", pady=2)
        self.worker_status = ttk.Label(status_frame, text="● Worker'ы не запущены", foreground="red")
        self.worker_status.pack(anchor="w", pady=2)
        self.progress_var = tk.StringVar(value="")
        self.progress_label = ttk.Label(status_frame, textvariable=self.progress_var, foreground="blue")
        self.progress_label.pack(anchor="w", pady=2)

        # Управление worker'ами
        worker_frame = ttk.LabelFrame(left_frame, text="Управление worker'ами", padding=10)
        worker_frame.pack(fill="x", padx=10, pady=5)
        self.btn_worker1 = ttk.Button(worker_frame, text="🖥️ Worker 5001", command=self.start_worker1, width=20)
        self.btn_worker1.pack(pady=2)
        self.btn_worker2 = ttk.Button(worker_frame, text="🖥️ Worker 5002", command=self.start_worker2, width=20)
        self.btn_worker2.pack(pady=2)
        self.btn_stop_workers = ttk.Button(worker_frame, text="⏹️ Остановить workers", command=self.stop_workers,
                                           width=20)
        self.btn_stop_workers.pack(pady=2)

        # Основные операции
        main_ops_frame = ttk.LabelFrame(left_frame, text="Основные операции", padding=10)
        main_ops_frame.pack(fill="x", padx=10, pady=5)
        self.btn_generate = ttk.Button(main_ops_frame, text="📁 1. Сгенерировать данные", command=self.generate_data,
                                       width=25)
        self.btn_generate.pack(pady=2)
        self.btn_master = ttk.Button(main_ops_frame, text="🚀 2. Запустить master.py", command=self.run_master, width=25)
        self.btn_master.pack(pady=2)
        self.btn_benchmark = ttk.Button(main_ops_frame, text="📊 3. Запустить benchmark.py", command=self.run_benchmark,
                                        width=25)
        self.btn_benchmark.pack(pady=2)

        # Модульные тесты
        test_frame = ttk.LabelFrame(left_frame, text="Модульные тесты", padding=10)
        test_frame.pack(fill="x", padx=10, pady=5)
        self.btn_test_orth = ttk.Button(test_frame, text="🧪 Тест MGS", command=self.run_test_orth, width=23)
        self.btn_test_orth.pack(pady=2)
        self.btn_test_io = ttk.Button(test_frame, text="🧪 Тест IO/Load", command=self.run_test_io, width=23)
        self.btn_test_io.pack(pady=2)
        self.btn_test_dist = ttk.Button(test_frame, text="🧪 Тест Distributed", command=self.run_test_dist, width=23)
        self.btn_test_dist.pack(pady=2)
        self.btn_test_all = ttk.Button(test_frame, text="🧪 Все тесты", command=self.run_all_tests, width=23)
        self.btn_test_all.pack(pady=2)

        # Кнопка остановки
        self.stop_button = ttk.Button(left_frame, text="⏹ ОСТАНОВИТЬ ВЫПОЛНЕНИЕ", command=self.stop_current_process,
                                      state="disabled")
        self.stop_button.pack(fill="x", padx=10, pady=10)
        ttk.Button(left_frame, text="Выход", command=self.on_close, width=15).pack(pady=5)

        # Правая панель – лог
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        log_frame = ttk.LabelFrame(right_frame, text="Лог выполнения (live)", padding=5)
        log_frame.pack(fill="both", expand=True)
        clear_btn_frame = ttk.Frame(log_frame)
        clear_btn_frame.pack(fill="x", pady=2)
        ttk.Label(clear_btn_frame, text="📋 Вывод программы:", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(clear_btn_frame, text="Очистить лог", command=self.clear_log).pack(side="right")
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 10), height=35, width=100)
        self.log_text.pack(fill="both", expand=True, pady=5)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------------- Логирование ----------------------
    def log(self, text):
        self.log_queue.put(text)

    def process_log_queue(self):
        try:
            while True:
                text = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, text + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_queue)

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        self.progress_var.set("")

    # ---------------------- Утилиты ----------------------
    def file_exists(self, filename):
        return os.path.exists(os.path.join(self.project_dir, filename))

    def get_subprocess_encoding(self):
        return "cp1251" if os.name == "nt" else "utf-8"

    def is_port_open(self, host, port, timeout=0.5):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    def workers_ready(self):
        return self.is_port_open("127.0.0.1", 5001) and self.is_port_open("127.0.0.1", 5002)

    def update_run_buttons(self):
        if self.workers_ready():
            self.btn_master.config(state="normal" if not self.is_running else "disabled")
            self.btn_benchmark.config(state="normal" if not self.is_running else "disabled")
            self.worker_status.config(text="● Worker'ы запущены", foreground="green")
        else:
            self.btn_master.config(state="disabled")
            self.btn_benchmark.config(state="disabled")
            self.worker_status.config(text="● Worker'ы не запущены", foreground="red")

        state = "disabled" if self.is_running else "normal"
        self.btn_generate.config(state=state)
        self.btn_worker1.config(state=state)
        self.btn_worker2.config(state=state)
        self.btn_stop_workers.config(state=state)
        self.btn_test_orth.config(state=state)
        self.btn_test_io.config(state=state)
        self.btn_test_dist.config(state=state)
        self.btn_test_all.config(state=state)
        self.root.after(2000, self.update_run_buttons)

    def set_running_state(self, running: bool):
        self.is_running = running
        if running:
            self.status_label.config(text="🔴 Выполняется...", foreground="orange")
            self.stop_button.config(state="normal")
        else:
            self.status_label.config(text="⚫ Готов к работе", foreground="green")
            self.stop_button.config(state="disabled")
            self.progress_var.set("")

    # ---------------------- Процессы ----------------------
    def stop_current_process(self):
        if self.current_process and self.current_process.poll() is None:
            try:
                if sys.platform == "win32":
                    self.current_process.terminate()
                else:
                    self.current_process.send_signal(signal.SIGTERM)
                self.log("\n[СТОП] Процесс остановлен пользователем")
            except Exception as e:
                self.log(f"[ОШИБКА] Не удалось остановить процесс: {e}")
        self.set_running_state(False)
        self.current_process = None

    def run_command_live(self, cmd, title=None, cwd=None):
        if self.is_running:
            messagebox.showwarning("Внимание", "Сначала дождитесь завершения текущей задачи или остановите её.")
            return

        def target():
            process = None
            try:
                self.set_running_state(True)
                if title:
                    self.log("=" * 80)
                    self.log(title)
                    self.log("=" * 80)
                    self.log("")
                self.log(f"[CMD] {' '.join(cmd)}\n")
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                process = subprocess.Popen(
                    cmd, cwd=cwd or self.project_dir,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env=env,
                    encoding=self.get_subprocess_encoding(), errors="replace"
                )
                self.current_process = process
                self.running_processes.append(process)
                for line in iter(process.stdout.readline, ''):
                    if line:
                        self.log(line.rstrip())
                        if "шаг" in line.lower() or "test" in line.lower() or "размерность" in line.lower():
                            self.progress_var.set(f"▶ {line[:50]}...")
                return_code = process.wait()
                self.log(f"\n[ЗАВЕРШЕНО] Код возврата: {return_code}" if return_code == 0 else f"\n[ОШИБКА] Код возврата: {return_code}")
            except Exception as e:
                self.log(f"[ОШИБКА] {e}")
            finally:
                if process in self.running_processes:
                    self.running_processes.remove(process)
                self.current_process = None
                self.set_running_state(False)

        threading.Thread(target=target, daemon=True).start()

    # ---------------------- Данные ----------------------
    def generate_data(self):
        if not self.file_exists("data_generator.py"):
            messagebox.showerror("Ошибка", "Файл data_generator.py не найден.")
            return
        n = self.n_var.get().strip()
        if not n.isdigit():
            messagebox.showerror("Ошибка", "Размерность n должна быть целым положительным числом.")
            return
        cmd = [self.python_exec, "data_generator.py", "--n", n]
        self.run_command_live(cmd, title=f"ГЕНЕРАЦИЯ ДАННЫХ (n={n})")

    def start_worker1(self):
        self._start_worker(5001, "WORKER-5001", "worker1_proc")

    def start_worker2(self):
        self._start_worker(5002, "WORKER-5002", "worker2_proc")

    def _start_worker(self, port, tag, proc_attr):
        if self.is_running:
            messagebox.showwarning("Внимание", "Дождитесь завершения текущей задачи.")
            return
        if not self.file_exists("worker.py"):
            messagebox.showerror("Ошибка", "Файл worker.py не найден.")
            return
        proc = getattr(self, proc_attr)
        if proc and proc.poll() is None:
            self.log(f"[INFO] {tag} уже запущен.")
            return
        try:
            proc = subprocess.Popen([self.python_exec, "worker.py", "--port", str(port)],
                                    cwd=self.project_dir,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True, bufsize=1,
                                    encoding=self.get_subprocess_encoding(),
                                    errors="replace")
            setattr(self, proc_attr, proc)
            self.log(f"[INFO] {tag} запущен.")
            threading.Thread(target=self._read_worker_output_live, args=(proc, tag), daemon=True).start()
        except Exception as e:
            self.log(f"[ОШИБКА] Не удалось запустить {tag}: {e}")

    def _read_worker_output_live(self, process, tag):
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log(f"[{tag}] {line.rstrip()}")
        except Exception:
            pass

    def stop_workers(self):
        stopped = False
        for i, proc in enumerate([self.worker1_proc, self.worker2_proc], 1):
            if proc and proc.poll() is None:
                proc.terminate()
                self.log(f"[INFO] Worker {5000+i} остановлен.")
                stopped = True
        if not stopped:
            self.log("[INFO] Нет активных worker-процессов.")

    # ---------------------- Master / Benchmark ----------------------
    def run_master(self):
        if not self.file_exists("master.py"):
            messagebox.showerror("Ошибка", "Файл master.py не найден.")
            return
        if not self.workers_ready():
            messagebox.showwarning("Worker-ы не запущены", "Сначала запустите Worker 5001 и Worker 5002.")
            return
        cmd = [self.python_exec, "master.py"]
        self.run_command_live(cmd, title="ЗАПУСК MASTER.PY")

    def run_benchmark(self):
        if not self.file_exists("benchmark.py"):
            messagebox.showerror("Ошибка", "Файл benchmark.py не найден.")
            return
        if not self.workers_ready():
            messagebox.showwarning("Worker-ы не запущены", "Сначала запустите Worker 5001 и Worker 5002.")
            return
        sizes = self.sizes_var.get().strip().split()
        dist_limit = self.dist_limit_var.get().strip()
        if not all(s.isdigit() for s in sizes) or not dist_limit.isdigit():
            messagebox.showerror("Ошибка", "Неверный формат sizes или dist-limit")
            return
        cmd = [self.python_exec, "benchmark.py", "--sizes"] + sizes + ["--dist-limit", dist_limit]
        self.run_command_live(cmd, title="ЗАПУСК BENCHMARK.PY (ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ)")

    # ---------------------- Тесты ----------------------
    def _run_pytest_file(self, test_file, title):
        if not self.file_exists(test_file):
            messagebox.showerror("Ошибка", f"Файл {test_file} не найден.")
            return
        cmd = [self.python_exec, "-m", "pytest", test_file, "-v"]
        self.run_command_live(cmd, title=title)

    def run_test_orth(self):
        self._run_pytest_file(os.path.join("tests", "test_orth_solver.py"), "МОДУЛЬНЫЙ ТЕСТ: MGS")

    def run_test_io(self):
        self._run_pytest_file(os.path.join("tests", "test_io_utils.py"), "МОДУЛЬНЫЙ ТЕСТ: IO/Load")

    def run_test_dist(self):
        self._run_pytest_file(os.path.join("tests", "test_distributed.py"), "МОДУЛЬНЫЙ ТЕСТ: Distributed")

    def run_all_tests(self):
        tests_dir = os.path.join(self.project_dir, "tests")
        if not os.path.exists(tests_dir):
            messagebox.showerror("Ошибка", "Папка tests не найдена.")
            return
        cmd = [self.python_exec, "-m", "pytest", tests_dir, "-v"]
        self.run_command_live(cmd, title="ЗАПУСК ВСЕХ МОДУЛЬНЫХ ТЕСТОВ")

    # ---------------------- Выход ----------------------
    def on_close(self):
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
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