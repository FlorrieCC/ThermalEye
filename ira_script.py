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
    # 设置是否保存数据（True：保存；False：不保存）
    save_flag = "False"  

    # 启动两个子进程，并传入 save_flag 参数
    process0 = subprocess.Popen(['python', 'ira/ira0.py', save_flag])
    process1 = subprocess.Popen(['python', 'ira/ira1.py', save_flag])
    
    print("程序已启动，按下 'q' 键将终止两个进程。")
    
    # 监听键盘输入，检测到 'q' 后发送 SIGINT 信号
    while True:
        ch = getch()
        if ch.lower() == 'q':
            print("检测到 'q'，正在发送中断信号...")
            process0.send_signal(signal.SIGINT)
            process1.send_signal(signal.SIGINT)
            break

    # 等待子进程退出
    process0.wait()
    process1.wait()
    print("所有进程已终止。")

if __name__ == "__main__":
    main()
