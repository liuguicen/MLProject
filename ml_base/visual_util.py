import cv2


def drawPoint(img, x, y):
    cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), 4)


def drawRectWh(img, x, y, w, h):
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 4)