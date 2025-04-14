import cv2 as cv
import numpy as np
import tkinter as tk
from time import sleep
from pynput import keyboard
import Quartz as quartz
from AppKit import NSWorkspace
from PIL import ImageGrab


class Capturer:
    def __init__(self):
        self.tkSelectionRoot = tk.Tk()

        self.screen = None
        self.screenLoopActive = False
        self.screenSize = {
            "x": 0,
            "y": 0,
            "width": self.tkSelectionRoot.winfo_screenwidth(),
            "height": self.tkSelectionRoot.winfo_screenheight()
        }

    def __captureScreen(self):
        self.screen = np.array(ImageGrab.grab(bbox=(
            int(self.screenSize["x"]), int(self.screenSize["y"]),
            int(self.screenSize["x"] + self.screenSize["width"]),
            int(self.screenSize["y"] + self.screenSize["height"]))))
        self.screen = cv.cvtColor(self.screen, cv.COLOR_RGB2BGR)
        return self.screen

    def __captureScreenLoop(self):
        while self.screenLoopActive:
            self.__captureScreen()
            sleep(0.05)
            if cv.waitKey(50) == ord('q'):
                self.stopCapture()

    def __waitForKey(self, key_char="c"):
        print(f"Presiona '{key_char}' para continuar...")

        key_pressed = {"pressed": False}

        def on_press(key):
            try:
                if key.char == key_char:
                    key_pressed["pressed"] = True
                    return False  # Detiene el listener
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    ## PUBLIC ##

    def selectWindow(self):
        print("Select the window to capture, then press c to select")
        self.__waitForKey("c")

        selectedPID = NSWorkspace.sharedWorkspace().activeApplication()['NSApplicationProcessIdentifier']
        options = quartz.kCGWindowListOptionOnScreenOnly
        windowList = quartz.CGWindowListCopyWindowInfo(options, quartz.kCGNullWindowID)
        for window in windowList:
            pid = window['kCGWindowOwnerPID']
            windowNumber = window['kCGWindowNumber']
            ownerName = window['kCGWindowOwnerName']
            geometry = window['kCGWindowBounds']
            windowTitle = window.get('kCGWindowName', u'Unknown')

            if selectedPID == pid:
                print(f"Window selected:\n{ownerName} - {windowTitle.encode('ascii','ignore')} (PID: {pid}, WID: {windowNumber}): {geometry}\n----------")
                self.screenSize = {
                    "x": geometry['X'],
                    "y": geometry['Y'],
                    "width": geometry['Width'],
                    "height": geometry['Height']
                }
        return [self.screenSize["x"], self.screenSize["y"],
                self.screenSize["x"] + self.screenSize["width"],
                self.screenSize["y"] + self.screenSize["height"]], self.__captureScreen().shape

    def stopCapture(self):
        self.screenLoopActive = False
        cv.destroyAllWindows()

    def startCapture(self):
        print("Screen capture started, press q (in the data window) to stop capturing")
        self.screenLoopActive = True
        self.__captureScreenLoop()

    def captureScreen(self):
        frame = self.__captureScreen()
        return frame
