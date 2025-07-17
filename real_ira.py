import subprocess
import sys
from datetime import datetime

def run_ira1_and_realsense(save_flag=True, run_time=0, ira1_dir="ira_data/", ira1_name="data", real_dir="real_data/", real_name="data"):
    save_flag_str = "True" if save_flag else "False"
    run_time_str = str(run_time)

    # Construct command for ira1.py
    ira1_cmd = [
        sys.executable,
        "ira/ira1.py",
        save_flag_str,
        run_time_str,
        ira1_dir,
        ira1_name
    ]

    # Construct command for realsense.py
    realsense_cmd = [
        sys.executable,
        "real/realsense.py",
        save_flag_str,
        run_time_str,
        real_dir,
        real_name
    ]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting ira1 subprocess:", " ".join(ira1_cmd))
    ira1_proc = subprocess.Popen(ira1_cmd)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting realsense subprocess:", " ".join(realsense_cmd))
    realsense_proc = subprocess.Popen(realsense_cmd)

    # Wait for both subprocesses to finish
    ira1_proc.wait()
    realsense_proc.wait()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Both subprocesses have finished.")

if __name__ == "__main__":
    run_ira1_and_realsense(
        save_flag=True,
        run_time=120,
        ira1_dir="ira_data/0430/",
        ira1_name="callibration",
        real_dir="real_data/0430/",
        real_name="callibration"
    )
