import cv2
import numpy as np
import os

from dataclasses import dataclass


# ------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------

@dataclass
class SelectedZone:
    p1: list
    p2: list
    sizeX: int
    sizeY: int
    totalPixels: int
    selectedPixels: np.ndarray


# ------------------------------------------------------------
# Global variables
# ------------------------------------------------------------

targetPath = "targets/"
resultsPath = "results/"

img = None
baseImage = None
selectedZone = None

drag = False
p1, p2 = [], []


# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------

def getTargetPicturesPaths():
    if not os.path.exists(targetPath):
        print(f'ERROR: can\'t find {targetPath} directory.')
        return

    files = os.listdir(targetPath)
    for i in range(len(files)):
        files[i] = targetPath + files[i]
    
    return files


def calculateRectangleData():
    global p1, p2

    if p1[0] > p2[0]:
        p1[0], p2[0] = p2[0], p1[0]
    if p1[1] > p2[1]:
        p1[1], p2[1] = p2[1], p1[1]

    sizeX = p2[0] - p1[0] + 1
    sizeY = p2[1] - p1[1] + 1
    totalPx = sizeX * sizeY

    print(f'Selected pixels: from {p1} to {p2}')
    print(f'Rectangle size: {sizeX} * {sizeY} ({totalPx} pixels)')

    cv2.destroyWindow("Selected pixels")
    selectedPixels = baseImage[p1[1]:p2[1], p1[0]:p2[0]]
    cv2.imshow("Selected pixels", selectedPixels)

    selectedZone = SelectedZone(p1, p2, sizeX, sizeY, totalPx, selectedPixels)


def mouseEvent(event, x, y, flags, params):
    global drag, p1, p2
    redraw = False

    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = [x, y]
        p2 = [x, y]
        drag = True
        redraw = True
    elif event == cv2.EVENT_LBUTTONUP:
        p2 = [x, y]
        drag = False
        redraw = True
        calculateRectangleData()
    
    if drag:
        p2 = [x, y]
        redraw = True
        
    if redraw:
        img = baseImage.copy()
        img = cv2.rectangle(img, tuple(p1), tuple(p2), (0, 0, 255), 1)
        cv2.imshow("First image", img)


def openFirstFile():
    global img, baseImage

    files = getTargetPicturesPaths()

    img = cv2.imread(files[0], cv2.IMREAD_COLOR)
    baseImage = img.copy()
    cv2.imshow("First image", img)

    cv2.setMouseCallback("First image", mouseEvent)


def waitForExit():
    # Wait for a key to be pressed to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n\n\n## BEGIN ##")
    openFirstFile()
    waitForExit()
    print("## END ##")


if __name__ == "__main__":
    main()
