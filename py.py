import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)  # You can change the argument to a video file name if you want to use a video file

# Initialize variables
ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(frame1_gray)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between frames
    frame_diff = cv2.absdiff(frame1_gray, frame2_gray)

    # Threshold the difference to get a binary image
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Update the mask
    mask = cv2.addWeighted(mask, 0.9, thresh, 0.1, 0)

    # Apply a colormap to the mask
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)

    # Display the heatmap
    cv2.imshow('Motion Heatmap', heatmap)

    # Update the previous frame
    frame1_gray = frame2_gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
