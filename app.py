import os
import sys
import socket
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import queue
import time
import signal


class SLAUApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распределённое решение СЛАУ")
        self.root.geometry("1400x800")  # Еще шире
        self.root.minsize(1200, 700)

        self.python_exec = sys.executable
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        self.worker1_proc = None
        self.worker2_proc = None

        # Текущий выполняемый процесс
        self.current_process = None
        self.is_running = False

        # Очереди для live-логирования
        self.log_queue = queue.Queue()
        self.running_processes = []

        self.create_widgets()

        # Запускаем проверку очереди логов
        self.process_log_queue()

        self.log("[INFO] Программа запущена. Готов к работе.")
        self.log("")

        self.update_run_buttons()

    def create_widgets(self):
        # Основной контейнер с двумя панелями
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # ===== ЛЕВАЯ ПАНЕЛЬ ===== (кнопки и настройки)
        left_frame = ttk.Frame(main_paned, width=450)
        main_paned.add(left_frame, weight=1)

        # Заголовок
        title = ttk.Label(
            left_frame,
            text="Распределённое решение СЛАУ",
            font=("Segoe UI", 12, "bold")
        )
        title.pack(pady=10)

        # === Параметры ===
        params_frame = ttk.LabelFrame(left_frame, text="Параметры", padding=10)
        params_frame.pack(fill="x", padx=10, pady=5)

        # Размерность n
        n_frame = ttk.Frame(params_frame)
        n_frame.pack(fill="x", pady=2)
        ttk.Label(n_frame, text="Размерность n:", width=15).pack(side="left")
        self.n_var = tk.StringVar(value="100")
        ttk.Entry(n_frame, textvariable=self.n_var, width=15).pack(side="left", padx=5)

        # Benchmark sizes
        sizes_frame = ttk.Frame(params_frame)
        sizes_frame.pack(fill="x", pady=2)
        ttk.Label(sizes_frame, text="Benchmark sizes:", width=15).pack(side="left")
        self.sizes_var = tk.StringVar(value="10 30 50 100 300 1000 2000 5000")
        ttk.Entry(sizes_frame, textvariable=self.sizes_var, width=30).pack(side="left", padx=5)

        # dist-limit
        dist_frame = ttk.Frame(params_frame)
        dist_frame.pack(fill="x", pady=2)
        ttk.Label(dist_frame, text="dist-limit:", width=15).pack(side="left")
        self.dist_limit_var = tk.StringVar(value="300")
        ttk.Entry(dist_frame, textvariable=self.dist_limit_var, width=15).pack(side="left", padx=5)

        # === Статус ===
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

        # === Управление worker'ами ===
        worker_frame = ttk.LabelFrame(left_frame, text="Управление worker'ами", padding=10)
        worker_frame.pack(fill="x", padx=10, pady=5)

        self.btn_worker1 = ttk.Button(worker_frame, text="🖥️ Worker 5001", command=self.start_worker1, width=20)
        self.btn_worker1.pack(pady=2)

        self.btn_worker2 = ttk.Button(worker_frame, text="🖥️ Worker 5002", command=self.start_worker2, width=20)
        self.btn_worker2.pack(pady=2)

        self.btn_stop_workers = ttk.Button(worker_frame, text="⏹️ Остановить workers", command=self.stop_workers,
                                           width=20)
        self.btn_stop_workers.pack(pady=2)

        # === Основные операции ===
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

        # === Модульные тесты ===
        test_frame = ttk.LabelFrame(left_frame, text="Модульные тесты", padding=10)
        test_frame.pack(fill="x", padx=10, pady=5)

        self.btn_test_gauss = ttk.Button(test_frame, text="🧪 Тест Gauss", command=self.run_test_gauss, width=23)
        self.btn_test_gauss.pack(pady=2)

        self.btn_test_orth = ttk.Button(test_frame, text="🧪 Тест Orth", command=self.run_test_orth, width=23)
        self.btn_test_orth.pack(pady=2)

        self.btn_test_back = ttk.Button(test_frame, text="🧪 Тест BackSub", command=self.run_test_back, width=23)
        self.btn_test_back.pack(pady=2)

        self.btn_test_io = ttk.Button(test_frame, text="🧪 Тест IO", command=self.run_test_io, width=23)
        self.btn_test_io.pack(pady=2)

        self.btn_test_dist = ttk.Button(test_frame, text="🧪 Тест Distributed", command=self.run_test_dist, width=23)
        self.btn_test_dist.pack(pady=2)

        self.btn_test_all = ttk.Button(test_frame, text="🧪 Все тесты", command=self.run_all_tests, width=23)
        self.btn_test_all.pack(pady=2)

        # Кнопка остановки выполнения
        self.stop_button = ttk.Button(left_frame, text="⏹ ОСТАНОВИТЬ ВЫПОЛНЕНИЕ", command=self.stop_current_process,
                                      state="disabled")
        self.stop_button.pack(fill="x", padx=10, pady=10)

        # Кнопка выхода
        ttk.Button(left_frame, text="Выход", command=self.on_close, width=15).pack(pady=5)

        # ===== ПРАВАЯ ПАНЕЛЬ ===== (лог на всю высоту)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)  # Лог занимает больше места

        # Лог выполнения
        log_frame = ttk.LabelFrame(right_frame, text="Лог выполнения (live)", padding=5)
        log_frame.pack(fill="both", expand=True)

        # Кнопка очистки лога
        clear_btn_frame = ttk.Frame(log_frame)
        clear_btn_frame.pack(fill="x", pady=2)

        ttk.Label(clear_btn_frame, text="📋 Вывод программы:", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(clear_btn_frame, text="Очистить лог", command=self.clear_log).pack(side="right")

        # Текстовое поле для лога - на всю высоту и ширину правой панели
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            height=35,  # Очень высокий
            width=100  # Широкий
        )
        self.log_text.pack(fill="both", expand=True, pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def log(self, text):
        """Добавление текста в лог (потокобезопасно)"""
        self.log_queue.put(text)

    def process_log_queue(self):
        """Обработка очереди логов в главном потоке"""
        try:
            while True:
                text = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, text + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            # Проверяем очередь каждые 100 мс
            self.root.after(100, self.process_log_queue)

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        self.progress_var.set("")

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
            self.btn_master.config(state="normal" if not self.is_running else "disabled")
            self.btn_benchmark.config(state="normal" if not self.is_running else "disabled")
            self.worker_status.config(text="● Worker'ы запущены", foreground="green")
        else:
            self.btn_master.config(state="disabled")
            self.btn_benchmark.config(state="disabled")
            self.worker_status.config(text="● Worker'ы не запущены", foreground="red")

        # Блокировка/разблокировка кнопок в зависимости от состояния выполнения
        state = "disabled" if self.is_running else "normal"
        self.btn_generate.config(state=state)
        self.btn_test_gauss.config(state=state)
        self.btn_test_orth.config(state=state)
        self.btn_test_back.config(state=state)
        self.btn_test_io.config(state=state)
        self.btn_test_dist.config(state=state)
        self.btn_test_all.config(state=state)
        self.btn_worker1.config(state=state)
        self.btn_worker2.config(state=state)
        self.btn_stop_workers.config(state=state)

        self.root.after(2000, self.update_run_buttons)

    def set_running_state(self, running: bool):
        """Установка состояния выполнения"""
        self.is_running = running
        if running:
            self.status_label.config(text="🔴 Выполняется...", foreground="orange")
            self.stop_button.config(state="normal")
        else:
            self.status_label.config(text="⚫ Готов к работе", foreground="green")
            self.stop_button.config(state="disabled")
            self.progress_var.set("")

    def stop_current_process(self):
        """Остановка текущего процесса"""
        if self.current_process and self.current_process.poll() is None:
            try:
                if sys.platform == "win32":
                    self.current_process.terminate()
                else:
                    self.current_process.send_signal(signal.SIGTERM)

                self.log("\n[СТОП] Процесс остановлен пользователем")
                self.set_running_state(False)
                self.current_process = None
            except Exception as e:
                self.log(f"[ОШИБКА] Не удалось остановить процесс: {e}")

    def run_command_live(self, cmd, title=None, cwd=None):
        """Запуск команды с live-логированием и блокировкой кнопок"""

        # Проверяем, не выполняется ли уже другая задача
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

                self.log(f"[CMD] {' '.join(cmd)}")
                self.log("")

                # Создаем окружение
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'

                # Запускаем процесс
                process = subprocess.Popen(
                    cmd,
                    cwd=cwd or self.project_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    encoding=self.get_subprocess_encoding(),
                    errors="replace"
                )

                self.current_process = process
                self.running_processes.append(process)

                # Читаем вывод построчно в реальном времени
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        self.log(line.rstrip())
                        # Обновляем индикатор
                        if "шаг" in line.lower() or "test" in line.lower() or "размерность" in line.lower():
                            self.progress_var.set(f"▶ {line[:50]}...")

                # Ждем завершения
                return_code = process.wait()

                self.log("")
                if return_code == 0:
                    self.log(f"[ЗАВЕРШЕНО] Код возврата: {return_code}")
                else:
                    self.log(f"[ОШИБКА] Код возврата: {return_code}")

            except Exception as e:
                self.log(f"[ОШИБКА] {e}")
            finally:
                if process in self.running_processes:
                    self.running_processes.remove(process)
                self.current_process = None
                self.set_running_state(False)

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
        self.run_command_live(cmd, title=f"ГЕНЕРАЦИЯ ДАННЫХ (n={n})")

    def start_worker1(self):
        if self.is_running:
            messagebox.showwarning("Внимание", "Дождитесь завершения текущей задачи.")
            return

        if not self.file_exists("worker.py"):
            messagebox.showerror("Ошибка", "Файл worker.py не найден.")
            return

        if self.worker1_proc and self.worker1_proc.poll() is None:
            self.log("[INFO] Worker 5001 уже запущен.")
            return

        try:
            self.worker1_proc = subprocess.Popen(
                [self.python_exec, "worker.py", "--port", "5001"],
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding=self.get_subprocess_encoding(),
                errors="replace"
            )
            self.log("[INFO] Worker 5001 запущен.")
            threading.Thread(target=self._read_worker_output_live,
                             args=(self.worker1_proc, "WORKER-5001"),
                             daemon=True).start()
        except Exception as e:
            self.log(f"[ОШИБКА] Не удалось запустить Worker 5001: {e}")

    def start_worker2(self):
        if self.is_running:
            messagebox.showwarning("Внимание", "Дождитесь завершения текущей задачи.")
            return

        if not self.file_exists("worker.py"):
            messagebox.showerror("Ошибка", "Файл worker.py не найден.")
            return

        if self.worker2_proc and self.worker2_proc.poll() is None:
            self.log("[INFO] Worker 5002 уже запущен.")
            return

        try:
            self.worker2_proc = subprocess.Popen(
                [self.python_exec, "worker.py", "--port", "5002"],
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding=self.get_subprocess_encoding(),
                errors="replace"
            )
            self.log("[INFO] Worker 5002 запущен.")
            threading.Thread(target=self._read_worker_output_live,
                             args=(self.worker2_proc, "WORKER-5002"),
                             daemon=True).start()
        except Exception as e:
            self.log(f"[ОШИБКА] Не удалось запустить Worker 5002: {e}")

    def _read_worker_output_live(self, process, tag):
        """Чтение вывода worker'а в реальном времени"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log(f"[{tag}] {line.rstrip()}")
        except Exception:
            pass

    def stop_workers(self):
        if self.is_running:
            messagebox.showwarning("Внимание", "Сначала остановите текущую задачу.")
            return

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

    def run_master(self):
        if not self.file_exists("master.py"):
            messagebox.showerror("Ошибка", "Файл master.py не найден.")
            return

        if not self.workers_ready():
            messagebox.showwarning(
                "Worker-ы не запущены",
                "Для запуска master.py необходимо сначала запустить Worker 5001 и Worker 5002."
            )
            return

        matrix_path = os.path.join(self.project_dir, "data", "matrix.txt")
        vector_path = os.path.join(self.project_dir, "data", "vector.txt")

        if not os.path.exists(matrix_path) or not os.path.exists(vector_path):
            messagebox.showerror(
                "Ошибка",
                "Файлы data/matrix.txt и/или data/vector.txt не найдены.\n"
                "Сначала выполните генерацию данных."
            )
            return

        cmd = [self.python_exec, "master.py"]
        self.run_command_live(cmd, title="ЗАПУСК MASTER.PY")

    def run_benchmark(self):
        if not self.file_exists("benchmark.py"):
            messagebox.showerror("Ошибка", "Файл benchmark.py не найден.")
            return

        if not self.workers_ready():
            messagebox.showwarning(
                "Worker-ы не запущены",
                "Для запуска benchmark.py необходимо сначала запустить Worker 5001 и Worker 5002."
            )
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

        cmd = [self.python_exec, "benchmark.py", "--sizes"] + size_parts + ["--dist-limit", dist_limit]
        self.run_command_live(cmd, title="ЗАПУСК BENCHMARK.PY (ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ)")

    def run_test_gauss(self):
        test_file = os.path.join("tests", "test_gauss_solver.py")
        if not self.file_exists(test_file):
            messagebox.showerror("Ошибка", f"Файл {test_file} не найден.")
            return

        cmd = [self.python_exec, "-m", "pytest", test_file, "-v"]
        self.run_command_live(cmd, title="МОДУЛЬНЫЙ ТЕСТ: МЕТОД ГАУССА")

    def run_test_orth(self):
        test_file = os.path.join("tests", "test_orth_solver.py")
        if not self.file_exists(test_file):
            messagebox.showerror("Ошибка", f"Файл {test_file} не найден.")
            return

        cmd = [self.python_exec, "-m", "pytest", test_file, "-v"]
        self.run_command_live(cmd, title="МОДУЛЬНЫЙ ТЕСТ: МЕТОД ОРТОГОНАЛИЗАЦИИ")

    def run_test_back(self):
        test_file = os.path.join("tests", "test_back_substitution.py")
        if not self.file_exists(test_file):
            messagebox.showerror("Ошибка", f"Файл {test_file} не найден.")
            return

        cmd = [self.python_exec, "-m", "pytest", test_file, "-v"]
        self.run_command_live(cmd, title="МОДУЛЬНЫЙ ТЕСТ: ОБРАТНАЯ ПОДСТАНОВКА")

    def run_test_io(self):
        test_file = os.path.join("tests", "test_io_utils.py")
        if not self.file_exists(test_file):
            messagebox.showerror("Ошибка", f"Файл {test_file} не найден.")
            return

        cmd = [self.python_exec, "-m", "pytest", test_file, "-v"]
        self.run_command_live(cmd, title="МОДУЛЬНЫЙ ТЕСТ: ВВОД-ВЫВОД")

    def run_test_dist(self):
        test_file = os.path.join("tests", "test_distributed.py")
        if not self.file_exists(test_file):
            messagebox.showerror("Ошибка", f"Файл {test_file} не найден.")
            return

        cmd = [self.python_exec, "-m", "pytest", test_file, "-v", "-m", "not integration"]
        self.run_command_live(cmd, title="МОДУЛЬНЫЙ ТЕСТ: РАСПРЕДЕЛЕНИЕ")

    def run_all_tests(self):
        tests_dir = os.path.join(self.project_dir, "tests")
        if not os.path.exists(tests_dir):
            messagebox.showerror("Ошибка", "Папка tests не найдена.")
            return

        cmd = [self.python_exec, "-m", "pytest", tests_dir, "-v"]
        self.run_command_live(cmd, title="ЗАПУСК ВСЕХ МОДУЛЬНЫХ ТЕСТОВ")

    def on_close(self):
        # Останавливаем все процессы при выходе
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