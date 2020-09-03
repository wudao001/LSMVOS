# LSMVOS: Long-Short-Term Similarity Matching for Video Object
## Zhang Xuerui, Yuan Xia
[arxiv](https://arxiv.org/abs/2009.00771)
## Overview
<img src="image/overview.png" width="400px"/>

## results
<table border="1">
<tr>
<td>Dataset</td>
<td>J&F↑</td>
<td>J mean↑</td>
<td>J recall↑</td>
<td>J decay↓</td>
<td>F mean↑</td>
<td>F recall↑</td>
<td>F decay↓</td>
<td>FPS↑</td>
</tr>
<tr>
<td>DAVIS 2016 val</td>
<td>86.5</td>
<td>85.7</td>
<td>97.1</td>
<td>5.1</td>
<td>87.3</td>
<td>96.1</td>
<td>4.9</td>
<td>21.3</td>
</tr>
</table>

<table border="1">
<tr>
<td>Dataset</td>
<td>J&F↑</td>
<td>J mean↑</td>
<td>J recall↑</td>
<td>J decay↓</td>
<td>F mean↑</td>
<td>F recall↑</td>
<td>F decay↓</td>
</tr>
<tr>
<td>DAVIS 2017 val</td>
<td>77.4</td>
<td>73.9</td>
<td>83.6</td>
<td>12.9</td>
<td>80.8</td>
<td>91.3</td>
<td>15.7</td>
</tr>
</table>

<table border="1">
<tr>
<td>Dataset</td>
<td>J&F↑</td>
<td>J mean↑</td>
<td>J recall↑</td>
<td>J decay↓</td>
<td>F mean↑</td>
<td>F recall↑</td>
<td>F decay↓</td>
</tr>
<tr>
<td>DAVIS 2017 test</td>
<td>67.4</td>
<td>63.7</td>
<td>72.7</td>
<td>16.9</td>
<td>71.2</td>
<td>81.4</td>
<td>16.5</td>
</tr>
</table>

## Download
[DAVIS](https://share.weiyun.com/nSPPQAV7)

## Visual
[DAVIS 2016 val](https://www.bilibili.com/video/BV1jK4y1Y7yd/)
<br/>
[DAVIS 2017 val](https://www.bilibili.com/video/BV1MC4y1t7R2/)
<br/>
[DAVIS 2017 test](https://www.bilibili.com/video/BV1Bh411d72y/)

## - Requirements
python ==3.6
torch ==1.3.1
torchvision == 0.4.2

The short-term matching module in LSMVOS uses the cross-correlation operation in Flownet2, and an implementation of pytorch needs to be [installed](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package). If it is training, you must install the Correlation module. If you are testing, you can use the local implementation. The local implementation will divide the picture into several groups, which can be flexibly adjusted according to the size of the GPU memory.

## - command
<code>
python evaluation.py --deviceID 0 1 --Using_Correlation False --scale 10 --batch_size 1 --year '2016' --mode 'val' --path 'LSMVOS_DAVIS_2016.pth' --root '/home/rv/Desktop/DAVIS'</br>
</code>
deviceID: GPUid，default is 0</br>
Using_Correlation: Whether Correlation is installed</br>
scale: Image grouping pixels， default 10 pixels in a column is 1 group</br>
batch_size: Batch size，default is 1</br>
year: Davis year，default is '2016'</br>
mode: Davis phase，default is 'val'</br>
path: 'Model path'</br>
root: 'Dateset root'</br>
