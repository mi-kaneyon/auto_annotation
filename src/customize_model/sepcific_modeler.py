import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader

# データセットの準備
transform = transforms.Compose([
    transforms.Resize((800, 800)),  # Fast R-CNNに合わせてサイズ調整
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root='./dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(root='./dataset/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# モデルの定義
def get_model(num_classes):
    # Fast R-CNNモデルのインスタンスを取得
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 分類器の入力特徴量数を取得
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # プリトレーニング済みのヘッドを新しいものに置き換え
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# データセットに応じてクラス数を調整（背景を含むため+1）
num_classes = len(train_dataset.classes) + 1
model = get_model(num_classes)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# トレーニング
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(torch.device("cuda")) for image in images)
        targets = [{k: v.to(torch.device("cuda")) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {losses.item()}')

# モデルの保存
torch.save(model.state_dict(), 'model_final.pth')
