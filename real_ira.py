import subprocess
import sys
import tty
import termios
import signal

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
    save_flag = "True"  # 若不保存数据，将此处改为 "False"

    # 启动三个子进程，并传入 save_flag 参数
    process0 = subprocess.Popen(['python', 'ira/ira0.py', save_flag])
    process1 = subprocess.Popen(['python', 'ira/ira1.py', save_flag])
    process2 = subprocess.Popen(['python', 'real/realsense.py', save_flag])
    
    print("所有程序已启动，按下 'q' 键将终止所有进程。")
    
    # 监听键盘输入，检测到 'q' 后发送 SIGINT 信号
    while True:
        ch = getch()
        if ch.lower() == 'q':
            print("检测到 'q'，正在发送中断信号...")
            process0.send_signal(signal.SIGINT)
            process1.send_signal(signal.SIGINT)
            process2.send_signal(signal.SIGINT)
            break

    # 等待所有子进程退出
    process0.wait()
    process1.wait()
    process2.wait()
    print("所有进程已终止。")

if __name__ == "__main__":
    main()
