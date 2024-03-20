import os
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torchvision.utils as vutils

# data expansion define
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

# raw data path
raw_data_dir = '../raw/'
image_files = [f for f in os.listdir(raw_data_dir) if os.path.isfile(os.path.join(raw_data_dir, f))]

# create directory
train_dir = '../processed/train/'
val_dir = '../processed/val/'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# listing expand data pair
extended_images_labels = []

# loading image and implement data expansion to add list
for file_name in image_files:
    image_path = os.path.join(raw_data_dir, file_name)
    image = Image.open(image_path).convert('RGB')  # RGB type
    for i in range(5):  # data expansion x5
        transformed_image = transform(image)
        # prefix name processing
        extended_label = f"{file_name}_ext_{i}.png"
        extended_images_labels.append((transformed_image, extended_label))

# separate training and validation data
train_data, val_data = train_test_split(extended_images_labels, test_size=0.2, random_state=42)

# saving train and val data
for image, label in train_data:
    save_path = os.path.join(train_dir, label)
    vutils.save_image(image, save_path)

for image, label in val_data:
    save_path = os.path.join(val_dir, label)
    vutils.save_image(image, save_path)

print("Ready data：Training and Validation data is saved.")
