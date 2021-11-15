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

# PATHS

targetPath = "targets/"
resultsPath = "results/"
files = [] # Paths of each target files


# FIRST IMAGE RELATED

img = None # Image shown in the main window
baseImage = None # Copy of that image, never modified, and copied to 'img' each time a draw call is made

drag = False
p1, p2 = [], []


# TARGET TRACKING RELATED

radius = 10

imgGray = None # Grayscale values of the current image, used for target tracking calculations

currentSelectedZone = None # Data associated with the selected zone in the current image
previousSelectedZone = None # Data associated with the selected zone in the previous image

possibleStartingCoordinates = [] # Each possible p1 for the currentSelectedZone


# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------

def generateTargetPicturesPaths():
    global files

    if not os.path.exists(targetPath):
        print(f'ERROR: can\'t find {targetPath} directory.')
        return

    files = os.listdir(targetPath)
    for i in range(len(files)):
        files[i] = targetPath + files[i]


def calculateRectangleData():
    global imgGray, currentSelectedZone, p1, p2, currentSelectedZone

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
    selectedPixels = imgGray[p1[1]:p2[1], p1[0]:p2[0]]
    cv2.imshow("Selected pixels", selectedPixels)

    currentSelectedZone = SelectedZone(p1, p2, sizeX, sizeY, totalPx, selectedPixels)


def handleMouseEvents(event, x, y, flags, params):
    global img, baseImage, drag, p1, p2
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
        if p1[0] != p2[0] and p1[1] != p2[1]:
            calculateRectangleData()
    
    if drag:
        p2 = [x, y]
        redraw = True
        
    if redraw:
        img = baseImage.copy()
        if p1[0] != p2[0] and p1[1] != p2[1]:
            img = cv2.rectangle(img, tuple(p1), tuple(p2), (0, 0, 255), 1)
        cv2.imshow("First image", img)


def openFirstFile():
    global img, baseImage, imgGray, files

    generateTargetPicturesPaths()

    img = cv2.imread(files[0], cv2.IMREAD_COLOR)
    baseImage = img.copy()
    imgGray = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    cv2.imshow("First image", img)

    cv2.setMouseCallback("First image", handleMouseEvents)


def generatePossibleStartingCoordinates(height, width):
    global possibleStartingCoordinates, previousSelectedZone

    possibleStartingCoordinates = []

    previousStartX = previousSelectedZone.p1[0]
    previousStartY = previousSelectedZone.p1[1]
    for y in range(previousStartY - radius, previousStartY + radius + 1):
        if 0 <= y <= height - previousSelectedZone.sizeY:
            for x in range(previousStartX - radius, previousStartX + radius + 1):
                if 0 <= x <= width - previousSelectedZone.sizeX:
                    possibleStartingCoordinates.append((x, y))


def trackTarget():
    global currentSelectedZone, previousSelectedZone, files
    print("Rendering target tracking...")

    if currentSelectedZone == None:
        print("No zone selected. Please select a zone you want to track in the next images first.")
        return
    
    for i in range(1, len(files) - 1):
        previousSelectedZone = currentSelectedZone
        currentSelectedZone = None

        currentImageGray = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        height = currentImageGray.shape[0]
        width = currentImageGray.shape[1]

        generatePossibleStartingCoordinates(height, width)
        print(f'From p1 = {p1}')
        print(f'With image size = [{width} * {height}]')
        print(f'And selection size = [{previousSelectedZone.sizeX} * {previousSelectedZone.sizeY}]')
        print("Possible starting coordinates:")
        for coords in possibleStartingCoordinates:
            print(coords)
        break


def handleKeyboardEvents():
    while cv2.getWindowProperty("First image", 0) >= 0:
        key = cv2.waitKey(0)
        # See https://www.asciitable.com/ for ASCII codes
        if key == 27: # ESCAPE
            break
        elif key == ord('r'):
            trackTarget()
            
    cv2.destroyAllWindows()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n\n\n## BEGIN ##")
    openFirstFile()
    handleKeyboardEvents()
    print("## END ##")


if __name__ == "__main__":
    main()
