import cv2
import numpy as np
import torch
from model import CNN


# LOAD MODEL

device = torch.device("cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("MNIST_CNN_Model.pth", map_location=device))
model.eval()


# PREDICTION FUNCTION

def predict_digit(canvas):

    # Convert to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Find non-zero pixels (digit area)
    coords = cv2.findNonZero(gray)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = gray[y:y+h, x:x+w]
    else:
        digit = gray

    # Resize to 28x28
    digit = cv2.resize(digit, (28, 28))

    # Normalize (same as training)
    digit = digit / 255.0

    # Convert to tensor
    tensor = torch.tensor(digit, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)

    return pred.item()


# LOAD HSV VALUES

loadFromSys = True

if loadFromSys:
    hsv_value = np.load('hsv_value.npy')

# 
# CAMERA SETUP

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5, 5), np.uint8)

canvas = None
x1, y1 = 0, 0
noise_thresh = 800

predicted_digit = None


# MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if loadFromSys:
        lower_range = hsv_value[0]
        upper_range = hsv_value[1]

    # Mask creation
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Noise removal
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            # 🔥 WHITE DRAWING (IMPORTANT)
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 8)

        x1, y1 = x2, y2
    else:
        x1, y1 = 0, 0

    # Overlay drawing
    frame = cv2.add(frame, canvas)

    # Show prediction on screen
    if predicted_digit is not None:
        cv2.putText(frame, f"Digit: {predicted_digit}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3)

    # Display
    stacked = np.hstack((canvas, frame))
    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    key = cv2.waitKey(1)

    # EXIT (Enter key)
    if key == 10:
        break

    # CLEAR
    if key & 0xFF == ord('c'):
        canvas = None
        predicted_digit = None

    # PREDICT
    if key & 0xFF == ord('p'):
        predicted_digit = predict_digit(canvas)
        print("Predicted Digit:", predicted_digit)


# CLEANUP

cap.release()
cv2.destroyAllWindows()