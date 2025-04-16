import subprocess
import sys
from datetime import datetime

def run_ira1_and_realsense(save_flag=True, run_time=0, ira1_dir="ira_data/", ira1_name="data", real_dir="real_data/", real_name="data"):
    save_flag_str = "True" if save_flag else "False"
    run_time_str = str(run_time)

    # 构造 ira1.py 命令
    ira1_cmd = [
        sys.executable,
        "ira/ira1.py",
        save_flag_str,
        run_time_str,
        ira1_dir,
        ira1_name
    ]

    # 构造 realsense.py 命令
    realsense_cmd = [
        sys.executable,
        "real/realsense.py",
        save_flag_str,
        run_time_str,
        real_dir,
        real_name
    ]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 启动 ira1 子进程：", " ".join(ira1_cmd))
    ira1_proc = subprocess.Popen(ira1_cmd)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 启动 realsense 子进程：", " ".join(realsense_cmd))
    realsense_proc = subprocess.Popen(realsense_cmd)

    # 等待两个子进程完成
    ira1_proc.wait()
    realsense_proc.wait()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 两个子进程已结束。")

if __name__ == "__main__":
    run_ira1_and_realsense(
        save_flag=True,
        run_time=125,
        ira1_dir="ira_data/0413/",
        ira1_name="cyh_2",
        real_dir="real_data/0413/",
        real_name="cyh_2"
    )
