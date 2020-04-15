# -*- coding: utf-8 -*-

from torchcmh.run import run
import torch

torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    run()

