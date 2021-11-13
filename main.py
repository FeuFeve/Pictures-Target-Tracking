import cv2
import numpy as np
import os


# ------------------------------------------------------------
# Global variables
# ------------------------------------------------------------

targetPath = "targets/"
resultsPath = "results/"

img = None
baseImage = None

drag = False
p1, p2 = [], []


# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------

def getTargetPicturesPaths():
    if not os.path.exists(targetPath):
        print("ERROR: can't find '" + targetPath + "' directory.")
        return

    files = os.listdir(targetPath)
    for i in range(len(files)):
        files[i] = targetPath + files[i]
    
    return files


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
        print("Selected pixels: " + str(tuple(p1)) + " to " + str(tuple(p2)))
    
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
