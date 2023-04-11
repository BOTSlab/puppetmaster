import cv2

def nothing(x):
    pass

class ManualController:
    def __init__(self):
        self.window_name = 'Manual Control'
        self.forward_trackbar = 'Forward Speed'
        self.angular_trackbar = 'Angular Speed'

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)

        cv2.createTrackbar(self.forward_trackbar, self.window_name, 0, 255, nothing)
        cv2.createTrackbar(self.angular_trackbar, self.window_name, 0, 5, nothing)
    
    def get_forward_angular_tuple(self):

        forward = cv2.getTrackbarPos(self.forward_trackbar, self.window_name)
        angular = cv2.getTrackbarPos(self.angular_trackbar, self.window_name)

        return forward, angular