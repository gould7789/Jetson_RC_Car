import evdev
import select

print("=== SEARCHING FOR REAL KEYBOARD ===")

# 1. Find all devices with 'MOSART' in name
all_devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
targets = [d for d in all_devices if "MOSART" in d.name]

if not targets:
    print("[ERROR] No 'MOSART' device found!")
    exit()

print(f"Found {len(targets)} devices with name 'MOSART'.")
device_map = {dev.fd: dev for dev in targets}

for dev in targets:
    # dev.fn is the path (e.g., /dev/input/eventX)
    print(f" -> Monitoring: {dev.name} (Path: {dev.fn})")

print("\n>>> PRESS ARROW KEYS NOW! <<<")

try:
    while True:
        r, w, x = select.select(device_map.keys(), [], [])
        
        for fd in r:
            dev = device_map[fd]
            for event in dev.read():
                # If key press detected (value 1)
                if event.type == evdev.ecodes.EV_KEY and event.value == 1:
                    print("\n-------------------------------------------")
                    print("!!! FOUND IT !!!")
                    print(f"Real Keyboard Path: {dev.fn}")
                    print(f"Key Code: {event.code}")
                    print("-------------------------------------------")

except KeyboardInterrupt:
    print("\nExiting...")