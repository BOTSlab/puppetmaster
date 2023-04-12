import cv2
from math import pi

def nothing(x):
    pass

class ManualController:
    def __init__(self):
        self.window_name = 'Manual Control'
        self.forward_trackbar = 'Forward Speed'
        self.angular_trackbar = 'Angular Speed'

        #cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)

        #cv2.createTrackbar(self.forward_trackbar, self.window_name, 0, 100, nothing)
        #cv2.createTrackbar(self.angular_trackbar, self.window_name, 0, 360, nothing)
    
    def get_forward_angular_tuple(self):

        #forward = cv2.getTrackbarPos(self.forward_trackbar, self.window_name)
        #angular = cv2.getTrackbarPos(self.angular_trackbar, self.window_name)
        c = cv2.waitKey(10)

        forward = 0
        angular = 0
        if c == -1:
            # No key pressed.
            pass
        elif c == 82:
            forward += 50
        elif c == 81:
            angular -= 0.3
        elif c == 83:
            angular += 0.3
        elif c == 27:
            quit()
        else:
            print(f"ManualController: Unknown key: {c}")
        print(f"forward, angular: {forward, angular}")

        return forward, angular