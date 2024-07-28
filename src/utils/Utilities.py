import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import ConcatDataset, Subset, DataLoader
import numpy as np
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets
from typing import Sequence, Union
import os
import torch
from torch import distributed as torch_dist
from torch import nn
import json
home = "/HOME/"

#For LARS Optimizer
EETA_DEFAULT = 0.001
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable

from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.folder import find_classes,make_dataset


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        batch_size = z_i.shape[0]

        # Concatenate the embeddings
        z = torch.cat((z_i, z_j), dim=0)

        # Compute similarity
        sim = (nn.CosineSimilarity(dim=2)(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature)
        # Compute labels
        labels = torch.arange(batch_size, device = z_i.device, dtype=torch.long)
        labels = labels.repeat(2)

        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, device = z_i.device, dtype=bool)
        # mask =torch.tensor(torch.diag(torch.ones(batch_size),batch_size)\
        #         + torch.diag(torch.ones(batch_size),-batch_size),dtype=bool)
        sim.masked_fill_(mask, -1e9)

        loss = self.criterion(sim, labels)
        return loss
    

class SimCLRLoss(torch.nn.Module):
    def __init__(self, batch_size: int, temperature: float):
        super(SimCLRLoss, self).__init__()
        self.xent_loss: NTXentLoss = NTXentLoss(temperature)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        return 0.5 * (self.xent_loss(z_i, z_j) + self.xent_loss(z_j, z_i))

class VideoSimCLRLoss:
    """class for computing the weighted Xent loss over 
    video/timeseries datasets for the SimCLR model."""

    def __init__(self, lambdas: Iterable, batch_size: int, temperature: float):
        """Computes the weighted Xent loss over video/timeseries 
        datasets for the SimCLR model."""
        # defining the underlying individual loss
        self.simclr_loss: SimCLRLoss = SimCLRLoss(temperature=temperature, batch_size=batch_size)

        # defining lambdas
        for v in lambdas:
            assert isinstance(v, int) or isinstance(v, float), "non-scalar lambda value(s)"
        self.lambdas: np.ndarray = np.asarray(lambdas).flatten()
        self.lambdas /= self.lambdas.sum()
        self._num_lambdas: int = self.lambdas.size

    def forward(self, *args)->float:
        """Computes the weighted video xent loss.
        
        Parameters
        ----------
        args:
            Image embeddings to compute the video loss over.

        Returns
        -------
        video_loss: float
            The weighted xent/SimCLR loss.
        """
        # checking correct length
        num_embeddings: int = len(args)
        exp_num_embeddings: int = self._num_lambdas+1 
        assert num_embeddings == exp_num_embeddings, f"Expected {exp_num_embeddings} embeddings of images, but {num_embeddings} given."

        # computing loss
        z0: torch.Tensor = args[0]
        args = args[1:]
        video_loss: float = 0
        for i, zi in enumerate(args):
            video_loss += self.lambdas[i] * self.simclr_loss(z0, zi)

        return video_loss

    def __call__(self, *args) -> float:
        """Computes the weighted video xent loss.
        
        Parameters
        ----------
        args:
            Pairs of image embeddings to compute the video loss over.

        Returns
        -------
        video_loss: float
            The weighted xent/SimCLR loss.
        """
        return self.forward(*args)


class VideoFrameDataset(Dataset):
    def __init__(self, root: str, transform, offsets: tuple = (1,)):
        self.trainset_augmented_1 = datasets.ImageFolder(
            root,
            transform=transform,
            target_transform=lambda t: 0,
        )
        self.trainset_augmented_2 = datasets.ImageFolder(
            root,
            transform=transform,
            target_transform=lambda t: 0,
        )

        self.offsets: tuple = offsets
        self._size: int = len(self.trainset_augmented_1)

    def __len__(self):
        return self._size
    
    def __getitem__(self, idx):
        # getting original pairs
        original_pair1 = self.trainset_augmented_1[idx]
        original_pair2 = self.trainset_augmented_2[idx]
        pairs = [original_pair1, original_pair2]

        # getting offset pairs
        for offset in self.offsets:
            # getting correct indices
            if idx + offset < self._size:
                offset_idx: int = idx + offset
            elif idx - offset >= 0:
                offset_idx: int = idx - offset
            else:
                raise IndexError(f"offset {offset} is too large for given dataset.")
            
            # getting offset pairs
            offset_pair = self.trainset_augmented_1[offset_idx]

            # appending
            pairs.append(offset_pair)
        return tuple(pairs)
    

class CustomVideoDataset(VisionDataset):
    """

    Args:
        root (str or ``pathlib.Path``): Root directory of the Dataset.
        frames_per_clip (int): number of frames in a clip.
        step_between_clips (int, optional): number of frames between each clip.
        fold (int, optional): which fold to use. Should be between 1 and 3.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that takes in a TxHxWxC video
            and returns a transformed version.
        output_format (str, optional): The format of the output video tensors (before transforms).
            Can be either "THWC" (default) or "TCHW".

    Returns:
        tuple: A 3-tuple with the following entries:

            - video (Tensor[T, H, W, C] or Tensor[T, C, H, W]): The `T` video frames
            -  audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
               and `L` is the number of points
            - label (int): class of the video clip
    """

    def __init__(
        self,
        root: Union[str, Path],
        frames_per_clip: int,
        step_between_clips: int = 1,
        frame_rate: Optional[int] = None,
        fold: int = 1,
        train: bool = True,
        transform: Optional[Callable] = None,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
        output_format: str = "TCHW",
    ) -> None:
        super().__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError(f"fold should be between 1 and 3, got {fold}")

        extensions = ("avi","mp4",)
        self.fold = fold
        self.train = train

        self.classes, class_to_idx = find_classes(self.root)
        self.samples = make_dataset(self.root, class_to_idx, extensions=extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            output_format=output_format,
        )
        # we bookkeep the full version of video clips because we want to be able
        # to return the metadata of full version rather than the subset version of
        # video clips
        self.full_video_clips = video_clips
        self.indices = self._select_fold(video_list, fold, train)
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.full_video_clips.metadata

    def _select_fold(self, video_list: List[str], fold: int, train: bool) -> List[int]:
        return range(len(video_list))

    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        # label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video1 = self.transform(video)
            video2 = self.transform(video)
        else:
            video1 = video
            video2 = video

        return video1.squeeze(), video2.squeeze()#, audio, label
    


class MixedCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.trainset_augmented_1 = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.trainset_augmented_2 = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    
    def __len__(self):
        return len(self.trainset_augmented_1)
    
    def __getitem__(self, idx):
        (image1, label1) = self.trainset_augmented_1[idx]
        (image2, label2) = self.trainset_augmented_2[idx]
        return ((image1, label1),(image2, label2))
    
class MixedImageNet(Dataset):
    def __init__(self, root, transform_normal=None, transform_augmented=None):
        self.trainset_augmented_1 = datasets.ImageNet(root=root, transform=transform_normal)
        self.trainset_augmented_2 = datasets.ImageNet(root=root, transform=transform_augmented)
    
    def __len__(self):
        return len(self.trainset_augmented_1)
    
    def __getitem__(self, idx):
        (image1, label1) = self.trainset_augmented_1[idx]
        (image2, label2) = self.trainset_augmented_2[idx]
        return ((image1, label1),(image2, label2))
    
class MixedINat(Dataset):
    def __init__(self, root, transform):
        self.trainset_augmented_1 = datasets.INaturalist(root=root, version='2021_train_mini', target_type = 'kingdom', transform=transform)
        self.trainset_augmented_2 = datasets.INaturalist(root=root, version='2021_train_mini',target_type = 'kingdom', transform=transform)
    
    def __len__(self):
        return len(self.trainset_augmented_1)
    
    def __getitem__(self, idx):
        (image1, label1) = self.trainset_augmented_1[idx]
        (image2, label2) = self.trainset_augmented_2[idx]
        return ((image1, label1),(image2, label2)) 
    
class MixedINat128(Dataset):
    def __init__(self, root, transform,transform2=None):
        self.trainset_augmented_1 = datasets.ImageFolder(
            root,
            transform=transform,
            # target_transform=lambda t: 0,
        )
        if transform2 == None:
            transform2 = transform
        self.trainset_augmented_2 = datasets.ImageFolder(
            root,
            transform=transform,
            # target_transform=lambda t: 0,
        )
    def __len__(self):
        return len(self.trainset_augmented_1)
    
    def __getitem__(self, idx):
        (image1, label1) = self.trainset_augmented_1[idx]
        (image2, label2) = self.trainset_augmented_2[idx]
        return ((image1, label1),(image2, label2))   
     
      
class MixedAnimalia(Dataset):
    def __init__(self, root, transform):
        self.trainset_augmented_1 = datasets.ImageFolder(
            root,
            transform=transform,
            target_transform=lambda t: 0,
        )
        self.trainset_augmented_2 = datasets.ImageFolder(
            root,
            transform=transform,
            target_transform=lambda t: 0,
        )
    def __len__(self):
        return len(self.trainset_augmented_1)
    
    def __getitem__(self, idx):
        (image1, label1) = self.trainset_augmented_1[idx]
        (image2, label2) = self.trainset_augmented_2[idx]
        return ((image1, label1),(image2, label2))   
    
class MixedPascal(Dataset):
    def __init__(self, root, transform):
        self.trainset_augmented_1 = datasets.VOCDetection(
                "/HOME/datasets/pascal_voc",
                download=True,
                transform=transform,
                target_transform=lambda t: 0,
            )
        self.trainset_augmented_2 = datasets.VOCDetection(
                "/HOME/datasets/pascal_voc",
                download=True,
                transform=transform,
                target_transform=lambda t: 0,
            )
    def __len__(self):
        return len(self.trainset_augmented_1)
    
    def __getitem__(self, idx):
        (image1, label1) = self.trainset_augmented_1[idx]
        (image2, label2) = self.trainset_augmented_2[idx]
        return ((image1, label1),(image2, label2))   
    
    

class OneHotEncode:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def __call__(self, mask):
        # Convert PIL image to numpy array
        mask_np = np.array(mask)
        
        # Initialize an empty tensor with shape (num_classes, height, width)
        one_hot_mask = torch.zeros((self.num_classes, *mask_np.shape[:2]), dtype=torch.float)
        
        # Apply one-hot encoding
        for i in range(self.num_classes):
            one_hot_mask[i] = torch.tensor((mask_np == (i+1)), dtype=torch.float)
        
        return one_hot_mask

def plot_loss(list, path, name_of_file):
    plt.figure()
    plt.plot(range(len(list)), list, c="blue", label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    # plt.legend() #Optional
    plt.savefig(f'{path}/{name_of_file}.png')
    plt.close()

def plot_accuracy(list, path, name_of_file):
    plt.figure()
    plt.plot(range(len(list)), list, c="blue", label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on validation set vs Epochs")
    # plt.legend() #Optional
    plt.savefig(f'{path}/{name_of_file}.png')
    plt.close()

def save_results(result_txt,value,model):
    file = "results.json"
    if not(os.path.exists(file)):
        data = {}
    else:
        with open(file, 'r') as f:
            data = json.load(f)
    if data.get(model) == None:
        data[model] = {}
    data[model][result_txt] = value
    with open(file, 'w') as file:
        json.dump(data, file, indent=4)

def plot_features(model, num_feats, dataloader,  epoch, val_df = None):
    preds = np.array([]).reshape((0,1))
    gt = np.array([]).reshape((0,1))
    feats = np.array([]).reshape((0,num_feats))
    model.eval()
    all_labels = []
    with torch.no_grad():
        for (x1, label),(x2,_) in dataloader:
            x1 = x1.squeeze().to(device = 'cuda:0', dtype = torch.float)#.view((-1,3,224,224))
            #y = y.to(device = 'cuda:0')#.view((-1,1))
            out = model.avgpool(model.resnet(x1)).squeeze()
            out = out.cpu().data.numpy()#.reshape((1,-1))
            feats = np.append(feats,out,axis = 0)
            all_labels.append(label)
    all_labels = torch.cat(all_labels)
    tsne = TSNE(n_components = 2, perplexity = 50)
    x_feats = tsne.fit_transform(feats)
    
    for i in torch.linspace(0,max(all_labels),10,dtype=int):
        #All labels has to be a number array so when doing all_labels==i we get the index
        plt.scatter(x_feats[all_labels==i,1][:128],x_feats[all_labels==i,0][:128])
    
    plt.legend([str(i.item()) for i in torch.linspace(0,max(all_labels),10,dtype=int)])
    plt.savefig(f'./figures/SimCLR_training_{epoch}.png') 
    plt.close()

def round_img(img, possible_values:torch.tensor):
    #normalize possible vals
    possible_values = (possible_values-possible_values.mean())/possible_values.std()
    mean = img.mean(dim=(2, 3)).squeeze()
    std = img.std(dim=(2, 3)).squeeze()
    img = (img-mean[:,None,None,None])/std[:,None,None,None]
    diff = torch.abs((img[:,:,:,:,None]-possible_values))
    return possible_values[torch.argmin(diff,dim=-1)]

def split_development_set(dev_set, batch_size):
    dev_list = []
    indices_dev_set = range(len(dev_set))

    split_dev = Subset(dev_set, indices_dev_set[0:int((len(dev_set)*(0.01)))])
    loader_split = DataLoader(split_dev, batch_size=batch_size, shuffle=True, num_workers=2)
    dev_list.append(loader_split)

    split_dev = Subset(dev_set, indices_dev_set[0:int((len(dev_set)*(0.1)))])
    loader_split = DataLoader(split_dev, batch_size=batch_size, shuffle=True, num_workers=2)
    dev_list.append(loader_split)

    split_dev = Subset(dev_set, indices_dev_set[0:int((len(dev_set)*(1)))])
    loader_split = DataLoader(split_dev, batch_size=batch_size, shuffle=True, num_workers=2)
    dev_list.append(loader_split)
    
    return dev_list

def accuracy(dataloader, model, device):
    model.eval()
    with torch.no_grad():
        acc = 0.0
        for i, (images, segmentations_true) in enumerate(dataloader, 1):
            images = images.to(device).float()
            segmentations_true = segmentations_true.to(device)
            segmentations_output = model(images.to(device)).cpu().detach()
            segmentations_true = torch.argmax(segmentations_true, dim=1,keepdim=True).detach().cpu()
            segmentations_output = torch.argmax(segmentations_output.squeeze(), dim=1,keepdim=True).detach().cpu()
            acc += ((segmentations_true==segmentations_output).sum()/len(segmentations_output.view(-1))).item()
        
        acc /= len(dataloader)
        return acc

def jacc_ind(dataloader, model, device):
    with torch.no_grad():
        jaccard_index = 0.0
        for i, (images, segmentations_true) in enumerate(dataloader, 1):
            images = images.to(device).float()
            segmentations_true = segmentations_true.to(device)
            segmentations_output = model(images.to(device)).cpu().detach()
            segmentations_true = torch.argmax(segmentations_true, dim=1, keepdim=True).detach().cpu()
            segmentations_output = torch.argmax(segmentations_output.squeeze(), dim=1, keepdim=True).detach().cpu()
            
            # Computing the intersection and union
            intersection = torch.logical_and(segmentations_true, segmentations_output).sum(dim=(1,2))
            union = torch.logical_or(segmentations_true, segmentations_output).sum(dim=(1,2))
            # Avoid division by zero
            mask = union != 0
            # Calculate Jaccard index for each batch
            jaccard_index += (intersection[mask] / union[mask]).sum().item()
        
        # Compute the average Jaccard index over all batches
        jaccard_index /= len(dataloader.dataset)
        
        return jaccard_index
    
def jaccard_index(dataloader, model, device):
    with torch.no_grad():
        jaccard_sum = 0.0
        num_samples = 0
        
        for images, segmentations_true in dataloader:
            images = images.to(device).float()
            segmentations_true = segmentations_true.to(device).long()
            
            segmentations_output = model(images)
            segmentations_output = torch.argmax(segmentations_output, dim=1)
            
            # Reshape the tensors if necessary
            if segmentations_true.shape != segmentations_output.shape:
                segmentations_true = segmentations_true.squeeze(1)
                segmentations_output = segmentations_output.squeeze(1)
            
            intersection = ((segmentations_true == segmentations_output).sum(dim=[1, 2])).float()
            union = ((segmentations_true != 0).sum(dim=[1, 2]) + (segmentations_output != 0).sum(dim=[1, 2]) - intersection).float()
            jaccard = intersection / union
            jaccard_sum += jaccard.mean().item()
            num_samples += 1
        
        return jaccard_sum / num_samples

def segmentation_plots(dataloader_batch, model, device, saving_path, name_of_file):
    fig,axs = plt.subplots(10,2,figsize=(4,20))
    # resnet.resnet.eval() SHOULD I STILL DO THIS?
    with torch.no_grad():
        images, segmentations = dataloader_batch
        images = images.to(device).float()
        model_out = model(images.to(device)).cpu().detach()
        segmentations = torch.argmax(segmentations, dim=1,keepdim=True).detach().cpu()
        labels = torch.argmax(model_out.squeeze(), dim=1,keepdim=True).detach().cpu()
        for i in range(10):
            axs[i,0].imshow(segmentations[i].squeeze().numpy())
            axs[i,1].imshow(labels[i].squeeze().numpy())
            axs[i,0].set_axis_off()
            axs[i,1].set_axis_off()
        fig.savefig(f'{saving_path}/{name_of_file}.png')
        plt.close(fig)
    return

def PET_dataset(size, batch_size):
    transform_PET = transforms.Compose([
    transforms.Resize(size=size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    target_transform_PET = transforms.Compose([
        transforms.Resize(size=size, interpolation=transforms.InterpolationMode.NEAREST),
        OneHotEncode(3)
    ])

    #Check for documentation: https://pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html
    #We need train, test and validation set. I use same percentages as CW1 for each
    full_PET = ConcatDataset([
        torchvision.datasets.OxfordIIITPet(root=f"{home}/simclr_pytorch/datasets", split="trainval", download=True, target_types="segmentation", transform=transform_PET, target_transform=target_transform_PET),
        torchvision.datasets.OxfordIIITPet(root=f"{home}/simclr_pytorch/datasets", split="test", download=True, target_types="segmentation", transform=transform_PET, target_transform=target_transform_PET)
    ])

    indices_full_PET = range(len(full_PET))
    development_set_PET = Subset(full_PET, indices_full_PET[0:int((len(full_PET)*0.8))])

    test_set_PET = Subset(full_PET, indices_full_PET[int((len(full_PET)/10)*8):-1])
    indices_development_set_PET = range(len(development_set_PET))
    train_set_PET = Subset(development_set_PET, indices_development_set_PET[0:int((len(development_set_PET)*0.9))])
    validation_set_PET = Subset(development_set_PET, indices_development_set_PET[int((len(development_set_PET)*0.9)):-1])

    #Dataloaders
    loader_train_PET = DataLoader(train_set_PET, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_validation_PET = DataLoader(validation_set_PET, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_test_PET = DataLoader(test_set_PET, batch_size=batch_size, shuffle=True, num_workers=2)

    #Special development (train+validation) dataloader
    #This is a list of dataloaders, one for each split, in our case: [1%, 10%, 100%]
    development_list_PET = split_development_set(
        dev_set=development_set_PET, 
        batch_size=batch_size
    )
    return loader_train_PET, loader_validation_PET, loader_test_PET, development_list_PET