import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# load to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading model
model_path = './model_final.pth'  # model saved path 
model = fasterrcnn_resnet50_fpn(weights=None)  # No use pretrained model
model.load_state_dict(torch.load(model_path))
model.eval()
model = model.to(device)  # model move to GPU

# setup camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Image transfer to GPU
    image = F.to_tensor(frame).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image)
    
    # CPU working for drawing
    predictions = {k: v.to('cpu') for k, v in predictions[0].items()}
    
    # predictions box and score
    boxes = predictions['boxes']
    scores = predictions['scores']
    
    # creating bounding box
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score > 0.8:  # high score
            color = (0, 255, 0)  # green
        elif score > 0.5:  # not sure
            color = (0, 0, 255)  # red
        else:
            continue  # low score is ignored
        
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        cv2.rectangle(frame, pt1, pt2, color, 2)
    
    # frame display
    cv2.imshow('frame', frame)
    
    # exit script by press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera and close window
cap.release()
cv2.destroyAllWindows()
