import os
import time

def run_script2():
    # Wait for one hour (3600 seconds)
    try:
        # Replace 'python' with the path to your Python executable if it's not in the system PATH.
        os.system('sudo python test.py --cfg /cpfs01/shared/public/Gzy/FTD-distillation/configs/TinyImageNet/ConvIN/IPC10.yaml')
        os.system('sudo python test.py --cfg /cpfs01/shared/public/Gzy/FTD-distillation/configs/TinyImageNet/ConvIN/IPC10_temp.yaml')
    except FileNotFoundError:
        print("Error: Python executable not found.")
    except subprocess.CalledProcessError:
        print("Error: Failed to run script2.py.")

if __name__ == "__main__":
    run_script2()
