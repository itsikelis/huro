import pyspacemouse
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MouseController:
    def __init__(self, device_info, name="spacemouse"):
        self.name = name
        self.dead_zone = 0.2
        try:
            self.spacemouse = pyspacemouse.open(device = device_info)
            print(
                f"{bcolors.OKGREEN}['MouseController] {self.name} Connected{bcolors.ENDC}",
                flush=True
            )
        except Exception as e:
            print(
                f"{bcolors.FAIL}[{self.name}] Error: {e}{bcolors.ENDC}",
                flush=True
            )

    def read_data(self):
        if self.spacemouse is None:
            return None

        state = self.spacemouse.read()
        if state is None:
            return None

        def dz(v):
            return 0.0 if abs(v) < self.dead_zone else v

        return {
            "x": round(dz(state.y), 3),
            "y": -round(dz(state.x), 3),
            "z": round(dz(state.z), 3),
            "roll": -round(getattr(state, "roll", 0.0), 3),
            "pitch": round(getattr(state, "pitch", 0.0), 3),
            "yaw": -round(getattr(state, "yaw", 0.0), 3),
            "left_button": state.buttons[0] if state.buttons else 0,
            "right_button": state.buttons[1] if state.buttons and len(state.buttons) > 1 else 0,
        }

def main():
    devices = pyspacemouse.list_devices()
    if not devices:
        print(f"{bcolors.WARNING}[3DMouseController] No 3D mouse devices found.{bcolors.ENDC}", flush=True)
        return
    
    right_device = 'SpaceMouse Wireless'

    controller = MouseController(device_info=right_device, name="right_spacemouse")
    if controller.spacemouse is None:
        return

    try:
        while True:
            state = controller.read_data()
            if state is not None:
                print(f"{bcolors.OKBLUE}[3DMouseController] {state}{bcolors.ENDC}", flush=True)
            time.sleep(0.001)
    except KeyboardInterrupt:
        print(f"{bcolors.WARNING}[3DMouseController] Exiting...{bcolors.ENDC}", flush=True)

if __name__ == "__main__":
    main()