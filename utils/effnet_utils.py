
# Libraries
# ---------------------------------
import cfg_effnet as cfg
import os
import timm
import time
import random
import math
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from contextlib import contextmanager
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Logging and reproducibility utilities
#-----------------------------------------------
#OUTPUT_DIR = './'

@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')

def init_logger(log_file='./train.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter("%(message)s"))
    handler2 = logging.FileHandler(filename=log_file)
    handler2.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=cfg.seed)

# Training metrics and time utilities
#---------------------------------------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

# Evaluation metrics
#------------------------------------------------------------------------------------
def plot_score(score_list):
    plt.plot(list(range(len(score_list))), score_list, '-o')
    plt.title('Score_plot')
    plt.xticks(list(range(len(score_list))))
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.plot();

def get_score(y_true, y_pred):
    '''return accuracy'''
    return accuracy_score(y_true, y_pred)

def get_auc(y_true, y_pred):
    '''return roc_auc'''
    auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
    return auc

# Data preprocessing and dataset
#----------------------------------------------------------------------------
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['labels'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = file_name
        image = cv2.imread(file_path)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(file_path)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image , label


def get_transforms(*, data, cfg=None):

    if data == 'train':
        return Compose([
           # Resize(cfg.size, cfg.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(), ])

    elif data == 'valid':
        return Compose([
          # Resize(cfg.size, cfg.size),
           Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
           ),
           ToTensorV2(),
       ])
# Training and validation functions
#---------------------------------------------------------------------------------------------
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   ))
    return losses.avg

def valid_fn(valid_loader, model, criterion, device, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.softmax(1).to('cpu').numpy())
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        batch_time.update(time.time() - end)
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

# Model Definition
#------------------------------------------------------------------------------------------------------------
class Model_type(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', pretrained=False, cfg=None):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=cfg.target_size)
        #self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        #x = self.out(self.dropout(x))
        x = self.model(x)
        return x
    
# Training loop
#-----------------------------------------------------------------------------------------------------------

def train_loop(train, val, cfg):

    LOGGER.info(f"========== start: training ==========")

    train_folds = train.reset_index(drop=True)
    valid_folds = val.reset_index(drop=True)

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train', cfg=cfg))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='valid', cfg=cfg))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    model = Model_type(cfg.model_name,cfg = cfg, pretrained=True)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.factor, patience=cfg.patience_lr, verbose=True, eps=cfg.eps)
    criterion = nn.CrossEntropyLoss()
    best_score = 0.
    best_loss = np.inf
    score_list = []
    patience_count = 0
    last_score = 0

    for epoch in range(cfg.epochs):
        start_time = time.time()
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg)
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device, cfg)

        valid_labels = val[cfg.target_col].values
        scheduler.step(avg_val_loss)
        accuracy = get_score(valid_labels, preds.argmax(1))
        val_pred = preds/preds.sum(axis=1)[:,None]
        score = get_auc(valid_labels, val_pred)
        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Accuracy: {accuracy} AUC: {score}')

        #Save best model
        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 'preds': preds}, os.path.join(cfg.output_dir, f'{cfg.model_name}_best.pth'))

        #Compute improvement on validation set
        score_list.append(score)
        if score > last_score: #the model is still improving
            last_score = score
            patience_count = 0 #restore the patience_count
        else: #the model does not improve for this epoch
            patience_count += 1
            print(f'{patience_count=}')
            if patience_count > cfg.patience_es:
                print(f'***Max patience count {patience_count} reached with AUC on valid set = {score:.4f}***')
                break

    if cfg.score_plot:
        plot_score(score_list)

    #check_point = torch.load(os.path.join(cfg.output_dir, f'{cfg.model_name}_best.pth'))
    check_point = torch.load(os.path.join(cfg.output_dir, f'{cfg.model_name}_best.pth'), weights_only=False)
    valid_folds['preds'] = check_point['preds'].argmax(1)

    return valid_folds


# main function
#--------------------------------------------------------------------------------------------------------

def main(cfg, train, val):

    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[cfg.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.5f}')

    oof_df = pd.DataFrame()

    _oof_df = train_loop(train, val, cfg)
    oof_df = pd.concat([oof_df, _oof_df])
    get_result(_oof_df)
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    oof_df.to_csv(cfg.output_dir+'oof_df.csv', index=False)

