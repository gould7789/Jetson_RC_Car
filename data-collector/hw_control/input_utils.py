import evdev
from evdev import ecodes
import select

# HardwareInput 클래스 (기존 잘 되던 코드 유지)
class HardwareInput:
    def __init__(self, device_name_part="MOSART"):
        self.device = None
        self._force_connect_event3()

    def _force_connect_event3(self):
        print(f"\n[System] '/dev/input/event3' (진짜 키보드)에 강제 연결 시도...")
        try:
            # 🔴 [핵심] 사용자가 확인한 'event3'로 무조건 직진!
            self.device = evdev.InputDevice('/dev/input/event3')
            
            print(f"[SUCCESS] 연결 성공!: {self.device.name}")
            print(f"          경로: {self.device.fn}")
            
        except Exception as e:
            print(f"[ERROR] 연결 실패: {e}")
            print(" -> 혹시 젯슨을 재부팅하셨나요? 재부팅하면 번호가 바뀔 수 있습니다.")
            self.device = None

    def read_loop(self):
        if self.device:
            return self.device.read_loop()
        else:
            return []

# run_inference.py와 연결하기 위한 함수
global_hw_input = None

def get_key_nonblock():
    global global_hw_input

    # 1. 처음 실행되면 HardwareInput 객체 생성 (연결 시도)
    if global_hw_input is None:
        global_hw_input = HardwareInput()
    
    # 연결 실패했으면 False 반환
    if global_hw_input.device is None:
        return False

    # 2. 논블로킹(0초 대기)으로 입력 확인
    try:
        r, w, x = select.select([global_hw_input.device.fd], [], [], 0.0)
        if not r:
            return False # 입력 없음

        # 3. 이벤트 읽어서 변환
        for event in global_hw_input.device.read():
            if event.type == ecodes.EV_KEY and event.value == 1: # 1 = Key Press
                
                if event.code == ecodes.KEY_UP:    return "UP"
                if event.code == ecodes.KEY_DOWN:  return "DOWN"
                if event.code == ecodes.KEY_LEFT:  return "LEFT"
                if event.code == ecodes.KEY_RIGHT: return "RIGHT"
                
                if event.code == ecodes.KEY_W:     return "UP"
                if event.code == ecodes.KEY_S:     return "s"
                if event.code == ecodes.KEY_A:     return "a"
                if event.code == ecodes.KEY_Z:     return "z"
                if event.code == ecodes.KEY_T:     return "t"
                if event.code == ecodes.KEY_Q:     return "q"
                if event.code == ecodes.KEY_ESC:   return "ESC"

    except Exception as e:
        print(f"[WARN] 키보드 읽기 에러: {e}")
        return False

    return False