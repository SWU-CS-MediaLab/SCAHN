# Self-Constraining and Attention-based Hashing Network for Bit-Scalable Cross-Modal Retrieval

## Abstract

Recently deep cross-modal hashing (CMH) have received increased attention in multimedia information retrieval, as it is able to combine the benefit from the low storage cost and search efficiency of hashing, and the strong capabilities of feature abstraction by deep neural networks. CMH can effectively integrates hash representation learning and hash function optimization into an end-to-end framework. However, most of existing deep cross-modal hashing methods use a one-size-fits-all high level representation resulting in the loss of the spatial information of data. Also, previous methods mostly generated fixed length hashing codes. Here, the signi cance level of different bits are equally weighted thereby restricting their practical exibility. To address these issues, we propose a self-constraining and attention-based hashing network (SCAHN) for bit-scalable cross-modal hashing. SCAHN integrates the label constraints from early and late-stages as well as their fused features into the hash representation and hash function learning. Moreover, as the fusion of early and late-stages features is based on an attention mechanism, each bit of the hashing codes can be unequally weighted so that the code lengths can be manipulated by ranking the signi cance of each bit without extra hash-model training. Extensive experiments conducted on four benchmark datasets demonstrate that our proposed SCAHN outperforms the current state-of-the-art CMH methods. Moreover, it is also shown that the generated bit-scalable hashing codes well-preserve the discriminative power with varying code lengths and obtain competitive results comparing to the state-of-the-art.

------

```latex
@article{SCAHN
author = {Xinzhi Wang,Xitao Zou,Erwin M. Bakker, Song Wu},
title = {Self-Constraining and Attention-based Hashing Network for Bit-Scalable Cross-Modal Retrieval},
journaltitle = {Neurocomputing},
date = {13 March 2020},
}
```

# deep cross modal hashing (torchcmh)

torchcmh is a library built on PyTorch for deep learning cross modal hashing.\
Including: 
- data visualization
- baseline methods
- multiple data reading API
- loss function API
- config call
----
### Dataset

There are four datasets(Mirflickr25k, Nus Wide, MS coco, IAPR TC-12) sort out by myself,
if you want use these datasets, please download mat file and image file by readme file in dataset package.\
Please read "[readme](./torchcmh/dataset/README.md)" in dataset package

----
### Model
You can crate model or use existing model. 
We support some pre-train models, you can check out the [README.md](./torchcmh/models/README.md) file in details.

---
### Dependencies 
you need to install these package to run
- visdom 0.1.8+
- pytorch 1.0.0+
- tqdm 4.0+
----
### Logs and checkpoints

All method training will create a log and checkpoint to store the model. \
you can find log in ./logs/\{method_name\}/\{dataset_name\}/date.txt \
you can find checkpoints in ./checkpoints/\{method_name\}/\{dataset_name\}/\{bit\}-\{model_name\}.pth

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

----
### How to create your method
- create new method file in folder ./torchcmh/training/
- inherit implement TrainBase
- change config.yml file and run.

#### some function in TrainBase
- loss variable \
In your method, some variables you need to store and check, such as loss and acc. 
You can use var in TrainBase "loss_store" to store:
```pythn
self.loss_store = ["log loss", 'quantization loss', 'balance loss', 'loss']
```
"loss_store" is a list, push the name and update value by "loss_store\[name\].update()":
```python
value = 1000    # the value to update 
n = 10          # the number of instance for current value
self.loss_store['log loss'].update(value, n)
```
For print and visualization the loss, you can use:
```python
epoch = 1       # current epoch
self.print_loss(epoch)  # print loss
self.plot_loss("img loss")  # visualization img loss is the name of chart
```
clean "loss_store"
```python
self.reset_loss()   # reset loss_store
```
- parameters
In your method, all parameters can be stored as follows:
```python
self.parameters = {'gamma': 1, 'eta': 1}    # {name: value}
```
when method training, log will record the parameters and learning rate.
- valid
```python
for epoch in range(self.max_epoch):
    # training codes
    self.valid(epoch)
```
----
### LICENSE
this repository keep MIT license.
