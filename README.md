# Self-Constraining and Attention-based Hashing Network for Bit-Scalable Cross-Modal Retrieval

## Abstract

Recently deep cross-modal hashing (CMH) have received increased attention in multimedia information retrieval, as it is able to combine the benefit from the low storage cost and search efficiency of hashing, and the strong capabilities of feature abstraction by deep neural networks. CMH can effectively integrates hash representation learning and hash function optimization into an end-to-end framework. However, most of existing deep cross-modal hashing methods use a one-size-fits-all high level representation resulting in the loss of the spatial information of data. Also, previous methods mostly generated fixed length hashing codes. Here, the signi cance level of different bits are equally weighted thereby restricting their practical exibility. To address these issues, we propose a self-constraining and attention-based hashing network (SCAHN) for bit-scalable cross-modal hashing. SCAHN integrates the label constraints from early and late-stages as well as their fused features into the hash representation and hash function learning. Moreover, as the fusion of early and late-stages features is based on an attention mechanism, each bit of the hashing codes can be unequally weighted so that the code lengths can be manipulated by ranking the signi cance of each bit without extra hash-model training. Extensive experiments conducted on four benchmark datasets demonstrate that our proposed SCAHN outperforms the current state-of-the-art CMH methods. Moreover, it is also shown that the generated bit-scalable hashing codes well-preserve the discriminative power with varying code lengths and obtain competitive results comparing to the state-of-the-art.

------

If you use this code as baseline, please refer the following paper:

@article{article,  
author = {Wang, Xinzhi and Zou, Xitao and Bakker, Erwin and Wu, Song},    
year = {2020},    
month = {03},    
pages = {},  
title = {Self-Constraining and Attention-based Hashing Network for Bit-Scalable Cross-Modal Retrieval},    
journal = {Neurocomputing},   
doi = {10.1016/j.neucom.2020.03.019}   
}  

---
### Dependencies 
you need to install these package to run
- visdom 0.1.8+
- pytorch 1.0.0+
- tqdm 4.0+  
- python 3.5+
----
### Logs and checkpoints

The training will create a log and checkpoint to store the model. \
You can find log in ./logs/\{method_name\}/\{dataset_name\}/date.txt \
You can find checkpoints in ./checkpoints/\{method_name\}/\{dataset_name\}/\{bit\}-\{model_name\}.pth

----
### How to using
- create a configuration file as ./script/default_config.yml
```yaml
training:
  # the name of python file in training
  method: SCAHN
  # the data set name, you can choose mirflickr25k, nus wide, ms coco, iapr tc-12
  dataName: Mirflickr25K
  batchSize: 64
  # the bit of hash codes
  bit: 64
  # if true, the program will be run on gpu. Of course, you need install 'cuda' and 'cudnn' better.
  cuda: True
  # the device id you want to use, if you want to multi gpu, you can use [id1, id2]
  device: 0
datasetPath:
  Mirflickr25k:
    # the path you download the image of data set. Attention: image files, not mat file.
    img_dir: \dataset\mirflickr25k\mirflickr

```
- run ./script/main.py and input configuration file path.
```python
from torchcmh.run import run
if __name__ == '__main__':
    run(config_path='default_config.yml')
```
- data visualization
Before you start trainer, please use command as follows to open visdom server.
```shell script
python -m visdom.server
```
Then you can see the charts in browser in special port.



