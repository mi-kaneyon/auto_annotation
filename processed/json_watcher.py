import json

# JSONファイルを読み込む
annotations_file = 'val_annotations.json'  # 適切なパスに置き換えてください

with open(annotations_file) as f:
    annotations = json.load(f)

# アノテーションデータからいくつかのバウンディングボックスを確認する
for annotation in annotations['annotations'][:10]:  # 最初の10個のアノテーションを表示
    bbox = annotation['bbox']
    x_min, y_min, width, height = bbox
    if width <= 0 or height <= 0:
        print(f"Invalid bbox found: {bbox} in image_id: {annotation['image_id']}")
    else:
        print(f"Valid bbox: {bbox} in image_id: {annotation['image_id']}")
