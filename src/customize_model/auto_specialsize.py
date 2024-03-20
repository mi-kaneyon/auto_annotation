import os
import json
import shutil
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.ops import box_iou
from sklearn.model_selection import train_test_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def load_model(model_path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def identify_special_person(crop_image_tensor, special_model, device):
    crop_image_tensor = crop_image_tensor.to(device)
    with torch.no_grad():
        prediction = special_model([crop_image_tensor])
    pred_labels = prediction[0]['labels']
    pred_scores = prediction[0]['scores']
    score_thresh = 0.5
    is_special_person = any((label == 1 and score > score_thresh) for label, score in zip(pred_labels, pred_scores))
    return is_special_person

def auto_annotate_and_draw(root_dir, subsets, resized_size=(768, 1028), score_thresh=0.8, iou_thresh=0.4, special_model_path='special.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    general_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    general_model.eval()
    special_model = load_model(special_model_path, device)

    for subset in subsets:
        images_dir = os.path.join(root_dir, 'processed', 'resized', subset)
        output_samples_dir = os.path.join(root_dir, 'processed', 'samples', subset)
        os.makedirs(output_samples_dir, exist_ok=True)
        images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        annotations = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}
        annotation_id = 1

        for image_id, img_name in enumerate(images):
            img_path = os.path.join(images_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(img).unsqueeze_(0).to(device)
            with torch.no_grad():
                general_pred = general_model(img_tensor)[0]

            pred_boxes = general_pred['boxes']
            pred_scores = general_pred['scores']
            keep = pred_scores >= score_thresh
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            keep_idxs = torchvision.ops.nms(pred_boxes, pred_scores, iou_thresh)
            final_boxes = pred_boxes[keep_idxs]
            final_scores = pred_scores[keep_idxs]

            for box, score in zip(final_boxes, final_scores):
                crop_image = img.crop((box[0].item(), box[1].item(), box[2].item(), box[3].item()))
                crop_image_tensor = F.to_tensor(crop_image).unsqueeze_(0)
                
                is_special_person = identify_special_person(crop_image_tensor, special_model, device)
                
                annotations["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1 if is_special_person else 2,
                    "category_name": "SpecialPerson" if is_special_person else "Person",
                    "bbox": [box[0].item(), box[1].item(), box[2].item() - box[0].item(), box[3].item() - box[1].item()],
                    "score": score.item(),
                    "iscrowd": 0
                })
                annotation_id += 1

            annotations["images"].append({
                "id": image_id,
                "file_name": img_name,
                "width": resized_size[0],
                "height": resized_size[1]
            })

            draw = ImageDraw.Draw(img)
            for box in final_boxes:
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            img.save(os.path.join(output_samples_dir, img_name))

        json_output_path = os.path.join(root_dir, 'processed', f"{subset}_annotations.json")
        with open(json_output_path, 'w') as f:
            json.dump(annotations, f)

def ensure_directories_exist(base_dir):
    dirs_to_create = [
        'processed/resized',
        'processed/resized/train',
        'processed/resized/val',
        'processed/train',
        'processed/val',
        'processed/samples',
        'processed/samples/train',
        'processed/samples/val'
    ]
    for dir in dirs_to_create:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(script_dir)
    ensure_directories_exist(project_root_dir)

    raw_dir = os.path.join(project_root_dir, 'raw')
    processed_dir = os.path.join(project_root_dir, 'processed')
    resized_dir = os.path.join(processed_dir, 'resized')
    train_dir = os.path.join(resized_dir, 'train')
    val_dir = os.path.join(resized_dir, 'val')

    resize_and_copy_images(raw_dir, resized_dir)
     split_data_and_save(resized_dir, train_dir, val_dir)
    auto_annotate_and_draw(project_root_dir, ['train', 'val'], special_model_path='special.pth')

def split_data_and_save(images_dir, train_dir, val_dir, test_size=0.2):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    train_images, val_images = train_test_split(images, test_size=test_size, random_state=42)
    for image in train_images:
        shutil.copy(os.path.join(images_dir, image), train_dir)
    for image in val_images:
        shutil.copy(os.path.join(images_dir, image), val_dir)

