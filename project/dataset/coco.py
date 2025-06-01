import os
import torch
import torchvision.transforms.v2
from torch.utils.data.dataset import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
import json
from collections import defaultdict


def load_coco_annotations(ann_file, img_dir):
    """
    Load COCO annotations from JSON file
    :param ann_file: Path to COCO annotations JSON file
    :param img_dir: Directory containing the images
    :return: List of image information dictionaries
    """
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # Create mappings for categories, images, and annotations
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    images = {img['id']: img for img in coco_data['images']}

    # Group annotations by image_id
    image_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        if ann['iscrowd'] == 0:  # Skip crowd annotations
            image_to_anns[ann['image_id']].append(ann)

    im_infos = []
    for img_id, img_anns in image_to_anns.items():
        if not img_anns:
            continue  # Skip images without annotations

        img_data = images[img_id]
        im_info = {
            'img_id': str(img_id),
            'filename': os.path.join(img_dir, img_data['file_name']),
            'width': img_data['width'],
            'height': img_data['height'],
            'detections': []
        }

        for ann in img_anns:
            # COCO bbox format is [x, y, width, height]
            # Convert to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            bbox = [x, y, x + w, y + h]

            # Skip tiny boxes
            if w < 1 or h < 1:
                continue

            det = {
                'label': ann['category_id'],
                'bbox': bbox,
                'difficult': 0  # COCO doesn't have this flag
            }
            im_info['detections'].append(det)

        # Only add images that have valid detections
        if im_info['detections']:
            im_infos.append(im_info)

    print(f'Total {len(im_infos)} images found')
    return im_infos


class COCODataset(Dataset):
    def __init__(self, split, coco_root, im_size=300):
        self.split = split
        self.coco_root = coco_root
        self.im_size = im_size
        self.im_mean = [123.0, 117.0, 104.0]
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

        # Set file paths based on split
        if split == 'train':
            self.ann_file = os.path.join(coco_root, 'annotations', 'instances_train2017.json')
            self.img_dir = os.path.join(coco_root, 'train2017')
        else:
            self.ann_file = os.path.join(coco_root, 'annotations', 'instances_val2017.json')
            self.img_dir = os.path.join(coco_root, 'val2017')

        # Train and test transformations (same as VOC)
        self.transforms = {
            'train': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.RandomPhotometricDistort(),
                torchvision.transforms.v2.RandomZoomOut(fill=self.im_mean),
                torchvision.transforms.v2.RandomIoUCrop(),
                torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.v2.Resize(size=(self.im_size, self.im_size)),
                torchvision.transforms.v2.SanitizeBoundingBoxes(
                    labels_getter=lambda transform_input:
                    (transform_input[1]["labels"], transform_input[1]["difficult"])),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                                                    std=self.imagenet_std)

            ]),
            'test': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.Resize(size=(self.im_size, self.im_size)),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                                                    std=self.imagenet_std)
            ]),
        }

        # Load category to index mapping
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)

        # Create index-to-label mapping
        # Start with background class
        self.idx2label = {0: 'background'}
        for cat in coco_data['categories']:
            self.idx2label[cat['id']] = cat['name']

        # Create label-to-index mapping
        self.label2idx = {v: k for k, v in self.idx2label.items()}

        # Load image info
        print(f"Loading {split} data...")
        self.images_info = load_coco_annotations(self.ann_file, self.img_dir)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        im_info = self.images_info[index]
        try:
            im = read_image(im_info['filename'])
        except Exception as e:
            print(f"Error loading image {im_info['filename']}: {e}")
            # Skip to the next index if there's an issue
            return self.__getitem__((index + 1) % len(self))

        # Get annotations for this image
        targets = {}
        targets['bboxes'] = tv_tensors.BoundingBoxes(
            [detection['bbox'] for detection in im_info['detections']],
            format='XYXY', canvas_size=im.shape[-2:])
        targets['labels'] = torch.as_tensor(
            [detection['label'] for detection in im_info['detections']])
        targets['difficult'] = torch.as_tensor(
            [detection['difficult'] for detection in im_info['detections']])

        # Transform the image and targets
        transformed_info = self.transforms[self.split](im, targets)
        im_tensor, targets = transformed_info

        h, w = im_tensor.shape[-2:]
        wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(targets['bboxes'])
        targets['bboxes'] = targets['bboxes'] / wh_tensor

        return im_tensor, targets, im_info['img_id']
