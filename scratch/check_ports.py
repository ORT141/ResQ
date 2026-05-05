import serial
import time

def check_port(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(0.5)
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            return data
        ser.close()
    except:
        pass
    return None

ports = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyAMA0']
bauds = [115200, 256000, 230400]

for p in ports:
    for b in bauds:
        print(f"Checking {p} at {b} baud...")
        data = check_port(p, b)
        if data:
            print(f"  [OK] Received {len(data)} bytes")
            print(f"  [DATA] {data[:50]}")
            if b'\xAA\xFF\x03\x00' in data:
                print("  [MATCH] Found LD2450 Header!")
            if b"heart rate =" in data or b"breath rate =" in data:
                print("  [MATCH] Found TickRix Text!")
        else:
            print(f"  [FAIL] No data")
