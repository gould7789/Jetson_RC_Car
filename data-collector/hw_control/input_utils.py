# -*- coding: utf-8 -*-
"""
===============================================================================
File Name    : input_utils.py
Description  : evdev를 사용하여 젯슨 나노에 연결된 하드웨어 키보드를 직접 제어
===============================================================================
"""
import evdev

class HardwareInput:
    def __init__(self, device_name_part="MOSART"):
        self.device = None
        self._force_connect_event3()

    def _force_connect_event3(self):
        print(f"\n[System] '/dev/input/event3' (진짜 키보드)에 강제 연결 시도...")
        try:
            # 🔴 [핵심] 아까 확인한 'event3'로 무조건 직진!
            self.device = evdev.InputDevice('/dev/input/event3')
            
            print(f"[SUCCESS] 연결 성공!: {self.device.name}")
            print(f"          경로: {self.device.fn}")
            
        except Exception as e:
            print(f"[ERROR] 연결 실패: {e}")
            print(" -> 혹시 젯슨을 재부팅하셨나요? 재부팅하면 번호가 바뀔 수 있습니다.")
            print(" -> 'find_real_keyboard.py'를 다시 실행해서 번호를 확인해주세요.")

    def read_loop(self):
        if self.device:
            return self.device.read_loop()
        else:
            return []