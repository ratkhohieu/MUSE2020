import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse

from metric import eval_metric
from model import *
from utils import *
import time
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

from warmup_scheduler import GradualWarmupScheduler

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_epochs', default=20, type=int, help='number epochs')
parser.add_argument('--num_classes', default=3, type=int, help='number classes')
parser.add_argument('--root_feature_folder', default='/media/sven/New Volume/features/c2_muse_topic/feature_segments'
                                                     '/egemaps_aligned/', type=str, help='root_feature_folder')
parser.add_argument('--root_label_folder', default='/media/sven/New Volume/features/c2_muse_topic'
                                                   '/label_segments/', type=str, help='root_label_folder')
parser.add_argument('--partition_info', default='/media/sven/New Volume/features/meta/processed_tasks/metadata'
                                                '/partition.csv', type=str, help='partition_info')
parser.add_argument('--label', default='valence', type=str, help='label')
parser.add_argument('--feature_set', default='xception', type=str, help='feature_set')

args = parser.parse_args()

device = "cuda:1" if torch.cuda.is_available() else 'cpu'


class Muse_Dataset(Dataset):
    def __init__(self, features, labels, id_segment):
        super().__init__()
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)
        self.id_segment = torch.from_numpy(id_segment)
        self.id_segment = self.id_segment.unsqueeze(-1)
        self.id_segment = self.id_segment.unsqueeze(-1).expand(-1, 16, -1)

        self.features = torch.cat((self.id_segment, self.features), dim=2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    dict_feature = prepare_data(args.root_feature_folder, args.root_label_folder,
                                args.label, args.partition_info, args.feature_set, mode='train')
    train_dataset = Muse_Dataset(dict_feature['train_feat'], dict_feature['train_lab'], dict_feature['train_id'])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True)

    dict_feature = prepare_data(args.root_feature_folder, args.root_label_folder,
                                args.label, args.partition_info, args.feature_set, mode='devel')
    valid_dataset = Muse_Dataset(dict_feature['devel_feat'], dict_feature['devel_lab'], dict_feature['devel_id'])
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False)

    torch.manual_seed(0)
    model = wavenet()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()

    cost_list = []

    scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0, last_epoch=-1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1,
                                              after_scheduler=scheduler_steplr)
    optimizer.zero_grad()
    optimizer.step()
    for epoch in tqdm(range(args.num_epochs), total=args.num_epochs, desc='Epoch'):
        model.train()
        print(epoch, optimizer.param_groups[0]['lr'])
        for batch_idx, (features, targets) in enumerate(train_loader):
            scheduler_warmup.step(epoch)

            features = features.to(device).float()
            targets = targets.to(device).long()
            targets = targets.squeeze(1)

            ### FORWARD AND BACK PROP
            out = model(features)

            loss = criterion(out, targets)

            optimizer.zero_grad()

            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            #################################################
            ### CODE ONLY FOR LOGGING BEYOND THIS POINT
            ################################################
            cost_list.append(loss.item())
            # if not batch_idx % 20:
            #     print(f'Epoch: {epoch + 1:03d}/{args.num_epochs:03d} | '
            #           f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
            #           f' Cost: {loss:.4f}')

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f' % (np.mean(cost_list)))

        model.eval()
        with torch.no_grad():
            pred = []
            targ = []
            for batch_idx, (features, targets) in enumerate(valid_loader):
                features = features.to(device).float()
                targets = targets.long()
                out = model(features)
                _, predicts = torch.max(out, 1)
                predicts = predicts.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
                pred.append(predicts)
                targ.append(np.concatenate(targets))
            targ = np.concatenate(targ)
            pred = np.concatenate(pred)
            eval_metric(pred, targ, 'arousal')
        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')

    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')


if __name__ == '__main__':
    main()
