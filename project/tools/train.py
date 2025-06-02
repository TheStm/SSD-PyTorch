import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
from model.ssd import SSD
import torchvision
from dataset.voc import VOCDataset
from dataset.coco import COCODataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def collate_function(data):
    return tuple(zip(*data))


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    #########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Determine which dataset to use based on config file
    dataset_type = 'voc'  # Default
    if args.config_path.endswith('coco.yaml'):
        dataset_type = 'coco'

    print(f"Using {dataset_type.upper()} dataset")

    # Create the appropriate dataset based on type
    if dataset_type == 'voc':
        train_dataset_obj = VOCDataset('train',
                              im_sets=dataset_config['train_im_sets'],
                              im_size=dataset_config['im_size'])
    else:  # COCO
        train_dataset_obj = COCODataset('train',
                               coco_root=dataset_config['coco_root'],
                               im_size=dataset_config['im_size'])

    train_dataset = DataLoader(train_dataset_obj,
                           batch_size=train_config['batch_size'],
                           shuffle=True,
                           collate_fn=collate_function)

    # Instantiate model and load checkpoint if present
    model = SSD(config=config['model_params'],
                num_classes=dataset_config['num_classes'])
    model.to(device)
    model.train()

    checkpoint_path = os.path.join(train_config['task_name'], f"{train_config['ckpt_name']}.pt")
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resuming from epoch {checkpoint['epoch']+1}")
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    if not os.path.exists(train_config['task_name']):
        os.makedirs(train_config['task_name'], exist_ok=True)

    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=model.parameters(),
                                weight_decay=5E-4, momentum=0.9)
    if start_epoch > 0 and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    lr_scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.5)

    # Skip epochs that have already been trained
    for _ in range(start_epoch):
        lr_scheduler.step()

    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0
    for i in range(start_epoch, num_epochs):
        ssd_classification_losses = []
        ssd_localization_losses = []
        for idx, (ims, targets, _) in enumerate(tqdm(train_dataset)):
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            images = torch.stack([im.float().to(device) for im in ims], dim=0)
            batch_losses, _ = model(images, targets)
            loss = batch_losses['classification']
            loss += batch_losses['bbox_regression']

            ssd_classification_losses.append(batch_losses['classification'].item())
            ssd_localization_losses.append(batch_losses['bbox_regression'].item())
            loss = loss / acc_steps
            loss.backward()

            if (idx + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if steps % train_config['log_steps'] == 0:
                loss_output = ''
                loss_output += 'SSD Classification Loss : {:.4f}'.format(np.mean(ssd_classification_losses))
                loss_output += ' | SSD Localization Loss : {:.4f}'.format(np.mean(ssd_localization_losses))
                print(loss_output)
            if torch.isnan(loss):
                print('Loss is becoming nan. Exiting')
                exit(0)
            steps += 1
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        print('Finished epoch {}'.format(i+1))
        loss_output = ''
        loss_output += 'SSD Classification Loss : {:.4f}'.format(np.mean(ssd_classification_losses))
        loss_output += ' | SSD Localization Loss : {:.4f}'.format(np.mean(ssd_localization_losses))
        print(loss_output)

        save_path = os.path.join(train_config['task_name'],
                                 f"{train_config['ckpt_name']}")

        checkpoint = {
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_output
        }

        torch.save(checkpoint, save_path)
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    train(args)

