import subprocess
import signal
import time

def main():
    # 设置是否保存数据：传入 "True" 或 "False"
    save_flag = "False"  # 若不保存数据，将此处改为 "False"
    run_time = "20"      # 以秒为单位的运行时间

    ira1_save_dir = "ira_data/0412/"
    ira1_save_name = "quick_2m"

    real_save_dir = "real_data/0412/"
    real_save_name = "output"

    try:
        run_time = float(run_time)
    except ValueError:
        print("运行时间参数不合法，使用默认无限制运行")
        run_time = 0

    # 启动子进程，并传入 save_flag 与 run_time 参数
    process0 = subprocess.Popen(
        ['python', 'ira/ira1.py', save_flag, str(run_time), ira1_save_dir, ira1_save_name],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0
    )
    # process1 = subprocess.Popen(
    #     ['python', 'ira/ira0.py', save_flag],
    #     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0
    # )
    process2 = subprocess.Popen(
        ['python', 'real/realsense.py', save_flag, str(run_time), real_save_dir, real_save_name],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0
    )

    print("所有程序已启动，等待设定运行时间结束将终止所有进程。")

    start_time = time.time()

    try:
        # 监听键盘输入，并检测是否达到设定的运行时长
        while True:
            # 检测运行时长，run_time 为正数时生效
            if run_time > 0 and (time.time() - start_time) > run_time:
                print(f"已运行 {run_time} 秒，达到设定运行时间，正在发送中断信号...")
                process0.terminate()
                # process1.terminate()
                process2.terminate()
                break

            time.sleep(0.1)  # 稍作等待，避免CPU占用过高
    finally:
        # 等待所有子进程退出
        process0.wait()
        # process1.wait()
        process2.wait()
        print("所有进程已终止。")

if __name__ == "__main__":
    # 为了使用 select 模块检测输入
    import select
    main()
