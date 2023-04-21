from torch.utils.data import DataLoader
import torchvision
from .datasets import CityFlowNLDataset, CityFlowNLInferenceDataset


def build_transform(is_train, cfg):
    if is_train:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((cfg.DATA.SIZE,cfg.DATA.SIZE)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(cfg.DATA.SIZE, scale=(0.8, 1.)),
            torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(10)],p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return transform

def build_dataloader(is_train, cfg):
    transform = build_transform(is_train, cfg)
    if not is_train:

        test_data=CityFlowNLInferenceDataset(cfg.DATA, transform=transform, type ='test')
        test_loader = DataLoader(dataset=test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=32)
        return test_loader
    
    else:
        train_data = CityFlowNLDataset(data_cfg=cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, transform=transform, type="train", finetune=cfg.MODEL.FINETUNE)
        train_loader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS)
    return train_loader