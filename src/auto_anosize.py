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

def auto_annotate_and_draw(root_dir, subsets, resized_size=(768, 1028), score_thresh=0.8, iou_thresh=0.4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loading model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    model.to(device)
    model.eval()


def resize_and_copy_images(source_dir, target_dir, size=(768, 1028)):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_dir, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size)
            img_resized.save(os.path.join(target_dir, filename))

def split_data_and_save(images_dir, train_dir, val_dir, test_size=0.2):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    train_images, val_images = train_test_split(images, test_size=test_size, random_state=42)
    for image in train_images:
        shutil.copy(os.path.join(images_dir, image), train_dir)
    for image in val_images:
        shutil.copy(os.path.join(images_dir, image), val_dir)

def auto_annotate_and_draw(root_dir, subsets, resized_size=(768, 1028), score_thresh=0.8, iou_thresh=0.4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()

    for subset in subsets:
        images_dir = os.path.join(root_dir, 'processed', 'resized', subset)
        output_samples_dir = os.path.join(root_dir, 'processed', 'samples', subset)
        os.makedirs(output_samples_dir, exist_ok=True)
        images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        annotations = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}
        annotation_id = 1

        for image_id, img_name in enumerate(images):  # loop image_id and img_name 
            img_path = os.path.join(images_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(img).unsqueeze_(0).to(device)
            with torch.no_grad():
                prediction = model(img_tensor)[0]

            # Pre-processfiltering for NMS
            pred_boxes = prediction['boxes']
            pred_scores = prediction['scores']
            keep = pred_scores >= score_thresh
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]

            # NMSimplementation
            keep_idxs = torchvision.ops.nms(pred_boxes, pred_scores, iou_thresh)

            # automatic bounding box
            final_boxes = pred_boxes[keep_idxs]
            final_scores = pred_scores[keep_idxs]

            for box, score in zip(final_boxes, final_scores):
                box = box.tolist()
                annotations["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # xmin, ymin, width, height 形式
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

    # ディレクトリ構造を確認し、必要に応じて作成
    ensure_directories_exist(project_root_dir)

    raw_dir = os.path.join(project_root_dir, 'raw')
    processed_dir = os.path.join(project_root_dir, 'processed')
    resized_dir = os.path.join(processed_dir, 'resized')
    train_dir = os.path.join(resized_dir, 'train')  # 更新: resized下にtrainとvalディレクトリを設定
    val_dir = os.path.join(resized_dir, 'val')      # 更新: resized下にtrainとvalディレクトリを設定

    # 以下の関数を実行する前に、上記でディレクトリを作成します
    resize_and_copy_images(raw_dir, resized_dir)
    split_data_and_save(resized_dir, train_dir, val_dir)
    auto_annotate_and_draw(project_root_dir, ['train', 'val'])
