import cv2
import numpy as np
from pathlib import Path
from train import LeNet5
import torch
from torchvision import transforms

model_path = Path(__file__).parent / "lenet5.pth"
if not model_path.exists():
    raise RuntimeError("Model not trained")

transform = transforms.Compose(
    (transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)))
    )

model = LeNet5()
model.load_state_dict(torch.load(model_path))
model.eval()

canvas = np.zeros((256, 256), dtype=np.uint8)

cv2.namedWindow("Canvas", cv2.WINDOW_GUI_NORMAL)
# cv2.namedWindow("Predict", cv2.WINDOW_GUI_NORMAL)

position = []
draw = False

def on_mouse(event, x, y, flags, param):
    global draw
    global position
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
    if event == cv2.EVENT_LBUTTONUP:
        draw = False
    if event == cv2.EVENT_MOUSEMOVE and draw:
        position = (y, x)

cv2.setMouseCallback("Canvas", on_mouse)

# while True:
#     if position:
#         cv2.circle(canvas, (position[1], position[0]), 5, (255), -1)
#     key = key = cv2.waitKey(10) & 0xFF
#     match key:
#         case 27:
#             break
#         case 8:
#             canvas *= 0
#         case 112:
#             with torch.no_grad():
#                 tensor = transform(canvas)
#                 batch = tensor.unsqueeze(0)
#                 cv2.imshow("Predict", batch[0].numpy().transpose((1, 2, 0)))
#                 output = model(batch)
#                 prediction = output.argmax(dim=1).item()
#                 probability = torch.softmax(output, dim=1).squeeze().cpu().numpy()
#                 print(prediction, (probability*100).astype("uint8"))


    # cv2.imshow("Canvas", canvas)


while True:
    if position:
        cv2.circle(canvas, (position[1], position[0]), 12, (255), -1)
    with torch.no_grad():
        tensor = transform(canvas)
        batch = tensor.unsqueeze(0)
        output = model(batch)
        prediction = output.argmax(dim=1).item()
        probability = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        # print(prediction, (probability*100).astype("uint8"))
        confidence = np.max(probability)*100
        display_canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        if confidence > 60:
            cv2.putText(display_canvas, f"{prediction} ({confidence:.1f}%)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_canvas, f"not sure ({confidence:.1f}%)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow("Canvas", display_canvas)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            break
        if key == 8:  # Backspace
            position = []
            canvas *= 0
cv2.destroyAllWindows()