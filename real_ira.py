import subprocess
import time
import threading
from datetime import datetime

def current_time_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 精确到毫秒

def start_process(cmd, name, creationflags):
    print(f"[{current_time_str()}] 启动{name}：{' '.join(cmd)}")
    return subprocess.Popen(cmd, creationflags=creationflags)

def main():
    save_flag = "False"
    run_time = "360"

    ira1_save_dir = "ira_data/0412/"
    ira1_save_name = "noise"

    real_save_dir = "real_data/0412/"
    real_save_name = "noise"

    try:
        run_time = float(run_time)
    except ValueError:
        print("运行时间参数不合法，使用默认无限制运行")
        run_time = 0

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0

    # 容器记录子进程
    processes = {}

    def launch_ira1():
        processes['ira1'] = start_process(
            ['python', 'ira/ira1.py', save_flag, str(run_time), ira1_save_dir, ira1_save_name],
            'ira1', creationflags
        )

    def launch_realsense():
        processes['realsense'] = start_process(
            ['python', 'real/realsense.py', save_flag, str(run_time), real_save_dir, real_save_name],
            'realsense', creationflags
        )

    # 使用线程“并发”启动
    thread1 = threading.Thread(target=launch_ira1)
    thread2 = threading.Thread(target=launch_realsense)

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    print("所有程序已启动。按 Ctrl+C 可提前终止运行。")

    start_time = time.time()

    try:
        while True:
            if run_time > 0 and (time.time() - start_time) > run_time:
                print(f"[{current_time_str()}] 已运行 {run_time} 秒，终止所有子进程...")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n[{current_time_str()}] 检测到 Ctrl+C，正在终止子进程...")
    finally:
        for name, p in processes.items():
            if p.poll() is None:
                print(f"[{current_time_str()}] 正在终止{name}")
                p.terminate()
        for name, p in processes.items():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[{current_time_str()}] 强制终止{name}")
                p.kill()

        print("所有子进程已终止。")

if __name__ == "__main__":
    main()
