import cv2
import numpy as np
import os


# ------------------------------------------------------------
# Global variables
# ------------------------------------------------------------

targetPath = "targets/"
resultsPath = "results/"


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


def openFirstFile():
    files = getTargetPicturesPaths()

    img = cv2.imread(files[0], cv2.IMREAD_COLOR)
    cv2.imshow("First image", img)

    # Wait for a key to be pressed to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n\n\n## BEGIN ##")
    openFirstFile()
    print("## END ##")


if __name__ == "__main__":
    main()
