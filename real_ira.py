import subprocess
import sys
import tty
import termios
import signal
import time

def getch():
    """获取单个字符输入，不需要按回车"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    # 设置是否保存数据：传入 "True" 或 "False"
    save_flag = "False"  # 若不保存数据，将此处改为 "False"
    run_time = "5"   # 以秒为单位的运行时间

    try:
        run_time = float(run_time)
    except ValueError:
        print("运行时间参数不合法，使用默认无限制运行")
        run_time = 0

    # 启动子进程，并传入 save_flag 与 run_time 参数
    process0 = subprocess.Popen(['python', 'ira/ira1.py', save_flag, str(run_time)])
    # process1 = subprocess.Popen(['python', 'ira/ira0.py', save_flag])
    # process2 = subprocess.Popen(['python', 'real/realsense.py', save_flag])
    
    print("所有程序已启动，按下 'q' 键或等待设定运行时间结束将终止所有进程。")
    
    start_time = time.time()
    
    # 监听键盘输入，并检测是否达到设定的运行时长
    while True:
        # 检测运行时长，run_time 为正数时生效
        if run_time > 0 and (time.time() - start_time) > run_time:
            print(f"已运行 {run_time} 秒，达到设定运行时间，正在发送中断信号...")
            process0.send_signal(signal.SIGINT)
            # process1.send_signal(signal.SIGINT)
            # process2.send_signal(signal.SIGINT)
            break
        
        # 检测键盘输入：按下 'q' 退出
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            ch = getch()
            if ch.lower() == 'q':
                print("检测到 'q'，正在发送中断信号...")
                process0.send_signal(signal.SIGINT)
                # process1.send_signal(signal.SIGINT)
                # process2.send_signal(signal.SIGINT)
                break
                
        time.sleep(0.1)  # 稍作等待，避免CPU占用过高

    # 等待所有子进程退出
    process0.wait()
    # process1.wait()
    # process2.wait()
    print("所有进程已终止。")

if __name__ == "__main__":
    # 为了使用 select 模块检测输入
    import select
    main()
