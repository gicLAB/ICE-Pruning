# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from functools import partial

DATA_BACKEND_CHOICES = ["pytorch", "syntetic"]
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types

    DATA_BACKEND_CHOICES.append("dali-gpu")
    DATA_BACKEND_CHOICES.append("dali-cpu")
except ImportError:
    print(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )


def load_jpeg_from_file(path, cuda=True, fp16=False):
    img_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    img = img_transforms(Image.open(path))
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        if fp16:
            mean = mean.half()
            std = std.half()
            img = img.half()
        else:
            img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input


#class HybridTrainPipe(Pipeline):
'''
    def __init__(
        self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False
    ):
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=rank,
            num_shards=world_size,
            random_shuffle=True,
        )

        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.ImageDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.ImageDecoder(
                device="mixed",
                output_type=types.RGB,
                device_memory_padding=211025920,
                host_memory_padding=140544512,
            )

        self.res = ops.RandomResizedCrop(
            device=dali_device,
            size=[crop, crop],
            interp_type=types.INTERP_LINEAR,
            random_aspect_ratio=[0.75, 4.0 / 3.0],
            random_area=[0.08, 1.0],
            num_attempts=100,
        )

        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            image_type=types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=rank,
            num_shards=world_size,
            random_shuffle=False,
        )

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size)
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            image_type=types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]
'''

class DALIWrapper(object):
    def gen_wrapper(dalipipeline, num_classes, one_hot, memory_format):
        for data in dalipipeline:
            input = data[0]["data"].contiguous(memory_format=memory_format)
            target = torch.reshape(data[0]["label"], [-1]).cuda().long()
            if one_hot:
                target = expand(num_classes, torch.float, target)
            yield input, target
        dalipipeline.reset()

    def __init__(self, dalipipeline, num_classes, one_hot, memory_format):
        self.dalipipeline = dalipipeline
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.memory_format = memory_format

    def __iter__(self):
        return DALIWrapper.gen_wrapper(
                self.dalipipeline, self.num_classes, self.one_hot, self.memory_format
        )


def get_dali_train_loader(dali_cpu=False):
    def gdtl(
        data_path,
        batch_size,
        num_classes,
        one_hot,
        start_epoch=0,
        workers=5,
        _worker_init_fn=None,
        fp16=False,
        memory_format=torch.contiguous_format,
    ):

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        traindir = os.path.join(data_path, "train")
        
        crop_size = 64 if "tiny-imagenet-200" in data_path else 224
        pipe = HybridTrainPipe(
            batch_size=batch_size,
            num_threads=workers,
            device_id=rank % torch.cuda.device_count(),
            data_dir=traindir,
            crop=crop_size,
            dali_cpu=dali_cpu,
        )
        print(f"Load training data from {traindir}")

        pipe.build()
        train_loader = DALIClassificationIterator(
            pipe, size=int(pipe.epoch_size("Reader") / world_size)
        )

        return (
            DALIWrapper(train_loader, num_classes, one_hot, memory_format),
            int(pipe.epoch_size("Reader") / (world_size * batch_size)),
        )

    return gdtl


def get_dali_val_loader():
    def gdvl(
        data_path,
        batch_size,
        num_classes,
        one_hot,
        workers=5,
        _worker_init_fn=None,
        fp16=False,
        memory_format=torch.contiguous_format,
    ):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        valdir = os.path.join(data_path, "val")

        crop_size = 64 if "tiny-imagenet-200" in data_path else 224
        img_size = 64 if "tiny-imagenet-200" in data_path else 256
        pipe = HybridValPipe(
            batch_size=batch_size,
            num_threads=workers,
            device_id=rank % torch.cuda.device_count(),
            data_dir=valdir,
            crop=crop_size,
            size=img_size,
        )
        print(f"Load testing data from {valdir}")

        pipe.build()
        val_loader = DALIClassificationIterator(
            pipe, size=int(pipe.epoch_size("Reader") / world_size)
        )

        return (
            DALIWrapper(val_loader, num_classes, one_hot, memory_format),
            int(pipe.epoch_size("Reader") / (world_size * batch_size)),
        )

    return gdvl


def fast_collate(memory_format, batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def expand(num_classes, dtype, tensor):
    e = torch.zeros(
        tensor.size(0), num_classes, dtype=dtype, device=torch.device("cuda")
    )
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e


class PrefetchedWrapper(object):
    def prefetched_loader(loader, num_classes, fp16, one_hot, data_path):
        mean = (
            torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )
        std = (
            torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )
        if fp16:
            mean = mean.half()
            std = std.half()

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if fp16:
                    next_input = next_input.half()
                    if one_hot:
                        next_target = expand(num_classes, torch.half, next_target)
                else:
                    next_input = next_input.float()
                    if one_hot:
                        next_target = expand(num_classes, torch.float, next_target)
                
                if "imagenet" in data_path:
                    next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, start_epoch, num_classes, fp16, one_hot, data_path):
        self.dataloader = dataloader
        self.fp16 = fp16
        self.epoch = start_epoch
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.data_path = data_path

    def __iter__(self):
        if self.dataloader.sampler is not None and isinstance(
            self.dataloader.sampler, torch.utils.data.distributed.DistributedSampler
        ):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(
            self.dataloader, self.num_classes, self.fp16, self.one_hot, self.data_path
        )

    def __len__(self):
        return len(self.dataloader)


def get_pytorch_train_loader(
    data_path,
    batch_size,
    num_classes,
    one_hot,
    start_epoch=0,
    workers=5,
    _worker_init_fn=None,
    fp16=False,
    memory_format=torch.contiguous_format,
):

    if "imagenet" in data_path: 
        traindir = os.path.join(data_path, "train")
        if "tiny-imagenet-200" in data_path:
            train_dataset = datasets.ImageFolder(traindir, transform=transforms.ToTensor())
            print(f"Load tiny-imagenet-200 training data from {traindir}")
        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
                ),
            )
            print(f"Load full-imagenet training data from {traindir}")
    elif "cifar10" in data_path:
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        T = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=T)
        print(f"Load cifar10")
    elif "mnist" in data_path:
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=T)
        print(f"=> Load MNIST training data, including {len(train_dataset)} (60,000) examples")
    else:
        raise EOFError("Cannot find the dataset")

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if "imagenet" in data_path: 
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=partial(fast_collate, memory_format),
            drop_last=True,
        )
    elif "cifar10" in data_path:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
    elif "mnist" in data_path:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )

    print(f"len(train_loader): {len(train_loader)}")

    return (
        PrefetchedWrapper(train_loader, start_epoch, num_classes, fp16, one_hot, data_path),
        len(train_loader),
    )


def get_pytorch_val_loader(
    data_path,
    batch_size,
    num_classes,
    one_hot,
    workers=5,
    _worker_init_fn=None,
    fp16=False,
    memory_format=torch.contiguous_format,
):
    if "imagenet" in data_path:
        valdir = os.path.join(data_path, "val")
        if "tiny-imagenet-200" in data_path:
            val_dataset = datasets.ImageFolder(valdir, transform=transforms.ToTensor())
            print(f"Load tiny-imagenet-200 testing data from {valdir}")
        else:
            val_dataset = datasets.ImageFolder(
                valdir, transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
            )
            print(f"Load full-imagenet testing data from {valdir}")
    elif "cifar10" in data_path:
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        T = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        val_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=T)
        print(f"Load cifar10 testing data")
    elif "mnist" in data_path:
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        val_dataset = datasets.MNIST(data_path, train=False, download=True, transform=T)
        print(f"Load MNIST testing data, including {len(val_dataset)} (10,000) examples")
    else:
        raise EOFError("Cannot find the dataset")

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    if "imagenet" in data_path:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=True,
            collate_fn=partial(fast_collate, memory_format),
        )
    elif "cifar10" in data_path:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=True,
            drop_last=True,
        )
    elif "mnist" in data_path:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=True,
            drop_last=True,
        )
    print(f"len(val_loader): {len(val_loader)}")
    return PrefetchedWrapper(val_loader, 0, num_classes, fp16, one_hot, data_path), len(val_loader)


class SynteticDataLoader(object):
    def __init__(
        self,
        fp16,
        batch_size,
        num_classes,
        num_channels,
        height,
        width,
        one_hot,
        memory_format=torch.contiguous_format,
    ):
        input_data = (
            torch.empty(batch_size, num_channels, height, width).contiguous(memory_format=memory_format).cuda().normal_(0, 1.0)
        )
        if one_hot:
            input_target = torch.empty(batch_size, num_classes).cuda()
            input_target[:, 0] = 1.0
        else:
            input_target = torch.randint(0, num_classes, (batch_size,))
        input_target = input_target.cuda()
        if fp16:
            input_data = input_data.half()

        self.input_data = input_data
        self.input_target = input_target

    def __iter__(self):
        while True:
            yield self.input_data, self.input_target


def get_syntetic_loader(
    data_path,
    batch_size,
    num_classes,
    one_hot,
    start_epoch=0,
    workers=None,
    _worker_init_fn=None,
    fp16=False,
    memory_format=torch.contiguous_format,
):
    return SynteticDataLoader(fp16, batch_size, num_classes, 3, 224, 224, one_hot, memory_format=memory_format), -1
