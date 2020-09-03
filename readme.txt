运行说明

需要安装：
python ==3.6
torch ==1.3.1
torchvision == 0.4.2

Davis数据集地址（https://davischallenge.org/index.html）

LSMVOS中短时匹配模块用到了Flownet2中的互相关运算，需要安装pytorch的一个实现（https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package）
如果是训练的话必须安装Correlation模块，如果测试的话可以使用本地实现，本地实现会将图片分成几组，可以根据显存大小灵活调整

指令：
python evaluation.py --deviceID 0 1 --Using_Correlation False --scale 10 --batch_size 1 --year '2016' --mode 'val' --path 'LSMVOS_DAVIS_2016.pth' --root '/home/rv/Desktop/DAVIS'

deviceID: 指GPUid，默认为0
Using_Correlation: 是否安装了Correlation模块
scale: 图片分组像素，默认按列10像素为1组
batch_size: 批量大小，默认为1
year: Davis数据集年份，默认为'2016'
mode: Davis数据集类别，默认为'val'
path: '模型参数地址'
root: '数据集地址'