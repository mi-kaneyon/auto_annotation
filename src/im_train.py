import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from PIL import Image
import json
import os


class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        with open(annotations_file) as f:
            annotations = json.load(f)
        self.img_annotations = annotations['annotations']
        self.imgs = {img['id']: img['file_name'] for img in annotations['images']}
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        img_info = self.img_annotations[idx]
        img_id = img_info['image_id']
        img_name = self.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        bbox = img_info['bbox']
        # transfer from json to model bounding box type
        x_min, y_min, w, h = bbox
        x_max = x_min + w
        y_max = y_min + h

        # modified bounding box
        boxes = torch.as_tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}

        if self.transform:
            image = self.transform(image)

        return image, target

def get_transform():
    return transforms.Compose([transforms.ToTensor()])

def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    # correting path
    train_dir = os.path.join(os.path.dirname(__file__), '..', 'processed', 'train')
    val_dir = os.path.join(os.path.dirname(__file__), '..', 'processed', 'val')
    train_annotations = os.path.join(os.path.dirname(__file__), '..', 'processed', 'train_annotations.json')
    val_annotations = os.path.join(os.path.dirname(__file__), '..', 'processed', 'val_annotations.json')

    # Dataset and DataLoader setting
    train_dataset = CustomDataset(train_annotations, train_dir, get_transform())
    val_dataset = CustomDataset(val_annotations, val_dir, get_transform())
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f'Epoch: {epoch+1}, Loss: {losses.item()}')

    # Save the model
    torch.save(model.state_dict(), '../src/model_final.pth')
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
