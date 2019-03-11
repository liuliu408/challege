# 智能盘点—钢筋数量AI识别(训练钢筋数据)

![图片标题](https://leanote.com/api/file/getImage?fileId=5c768143ab64415937002fe3)
**数据路径均是用的软连接方式！**
请提前把数据准备好，以及将数据集标注文件格式修改为COCO json格式！
本人是通过“ step_1C_voc_xml_json.py ” 代码实现的XML装COCO json格式

##1. 在 dataset_catalog.py 文件中设置指向自己的数据集
数据配置文件：这里的目的是指向训练数据集，以gj_2019_train为例，这里detectron本身的没有这个数据集的数据的，我们可以把自己做好的JPG文件夹放到_IM_DIR下面，把json文件放到_ANN_FN下面，以供后续使用。
 修改$~/detectron/detectron/datasets/dataset_catalog.py文件，在其中添加两个部分代码如下所示：
```
    'gj_2019_train': {
        _IM_DIR:
            _DATA_DIR + '/gj/train_dataset',
        _ANN_FN:
            _DATA_DIR + '/gj/annotations/gj_instances_train.json'
    },
    'gj_2019_test': {
        _IM_DIR:
            _DATA_DIR + '/gj/test_dataset',
        _ANN_FN:
            _DATA_DIR + '/gj/annotations/gj_instances_test.json'
    }
```
[生成训练 gj_instances_train.json](https://github.com/liuliu408/detect_steel_bar/blob/master/step1C_xml_json.py)
[生成测试 gj_instances_test.json](https://github.com/liuliu408/detect_steel_bar/blob/master/3image_2_json.py)

##2. 修改“ gj_1gpu_e2e_faster_rcnn_R-50-FPN.yaml ” 配置文件
你可以在detectron下面新建一个experiments的文件夹，用来存放这个yaml配置文件。为方便管理，我们通常将预训练模型pkl文件也放到此文件夹。注意每个yaml会对应一个pkl文件，自己想训练哪个模型，就去/detectron/configs路径找相应的yaml文件和去下载相应的pkl文件。yaml文件的修改如下面代码及其注释所示。

（文件路径：/home/qiang/detectron/config）
```
MODEL:
  TYPE: generalized_rcnn
  CONV_BODY: FPN.add_fpn_ResNet50_conv5_body
  NUM_CLASSES: 2   #自己训练类别+1
  FASTER_RCNN: True
NUM_GPUS: 1       #只有一个gpu就写1
SOLVER:
  WEIGHT_DECAY: 0.0001
  LR_POLICY: steps_with_decay
  BASE_LR: 0.0025
  GAMMA: 0.1
  MAX_ITER: 60000
  STEPS: [0, 30000, 40000]
  # Equivalent schedules with...
  # 1 GPU:
  #   BASE_LR: 0.0025
  #   MAX_ITER: 60000   #总的训练步数
  #   STEPS: [0, 30000, 40000]
  # 2 GPUs:
  #   BASE_LR: 0.005
  #   MAX_ITER: 30000
  #   STEPS: [0, 15000, 20000]
  # 4 GPUs:
  #   BASE_LR: 0.01
  #   MAX_ITER: 15000
  #   STEPS: [0, 7500, 10000]
  # 8 GPUs:
  #   BASE_LR: 0.02
  #   MAX_ITER: 7500
  #   STEPS: [0, 3750, 5000]
FPN:
  FPN_ON: True
  MULTILEVEL_ROIS: True
  MULTILEVEL_RPN: True
FAST_RCNN:
  ROI_BOX_HEAD: fast_rcnn_heads.add_roi_2mlp_head
  ROI_XFORM_METHOD: RoIAlign
  ROI_XFORM_RESOLUTION: 7
  ROI_XFORM_SAMPLING_RATIO: 2
TRAIN:
  #WEIGHTS: https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl  #预训练模型
  WEIGHTS: ./models/R-50.pkl
  DATASETS: ('gj_2019_train',)  #用到的训练数据集
  SCALES: (500,)
  MAX_SIZE: 2666
  BATCH_SIZE_PER_IM: 256
  RPN_PRE_NMS_TOP_N: 2000  # Per FPN level
TEST:
  DATASETS: ('gj_2019_test',)  #用到的测试数据集
  SCALE: 500
  MAX_SIZE: 2666
  NMS: 0.5
  RPN_PRE_NMS_TOP_N: 1000  # Per FPN level
  RPN_POST_NMS_TOP_N: 1000
OUTPUT_DIR: .   #训练结果存放位置
```

##3. 下载预训练模型
ImageNet Pretrained Models
The backbone models pretrained on ImageNet are available in the format used by Detectron. Unless otherwise noted, these models are trained on the standard ImageNet-1k dataset.
[R-50.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl): converted copy of MSRA's original ResNet-50 model
[R-101.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-101.pkl): converted copy of MSRA's original ResNet-101 model
[X-101-64x4d.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl): converted copy of FB's original ResNeXt-101-64x4d model trained with Torch7
[X-101-32x8d.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/20171220/X-101-32x8d.pkl): ResNeXt-101-32x8d model trained with Caffe2 at FB
[X-152-32x8d-IN5k.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl): ResNeXt-152-32x8d model trained on ImageNet-5k with Caffe2 at FB (see our ResNeXt paper for details on ImageNet-5k)

（1） ubuntu下下载预训练模型
 ubuntu下可以通过 **wget** 在model动物园里面下载所需要的模型文件！！
 [https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)
```
$ wget https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-50.pkl /tmp/detectron/detectron-download-cache 
/ImageNetPretrained/MSRA/R-50.pkl  
```
你也可以在训练的时候下载，不过估计没人这么干，肯定事先下载好。请新建 pre-trained_model（文件名自己随便命名）文件夹，把 R-50.pkl 模型文件放在该文件夹下！
（2）Windows下用迅雷下载模型，然后拷贝到ubuntu下的目录

##4. 单GPU训练
默认的单GPU是指定 “0号 GPU” 来训练模型
```
qiang@qiang:~/detectron$python tools/train_net.py \
--cfg configs/gj_1gpu_e2e_faster_rcnn_R-50-FPN.yaml \
OUTPUT_DIR /tmp/detectron-output
```
指定 “1,3号 GPU” 来训练模型
```
qiang@qiang:~/detectron$CUDA_VISIBLE_DEVICES='1,3' python tools/train_net.py \
--cfg configs/gj_2gpu_e2e_faster_rcnn_R-50-FPN.yaml \
OUTPUT_DIR /tmp/detectron-output
```
训练开始时间：2019-02-27 19:44:52  

训练结束返回数据
```
......限于篇幅，前面省略......
json_stats: {"accuracy_cls": "0.999023", "eta": "0:00:00", "iter": 59999, "loss": "0.031790", "loss_bbox": "0.014982", "loss_cls": "0.004914", "loss_rpn_bbox_fpn2": "0.010343", "loss_rpn_bbox_fpn3": "0.000111", "loss_rpn_bbox_fpn4": "0.000000", "loss_rpn_bbox_fpn5": "0.000000", "loss_rpn_bbox_fpn6": "0.000000", "loss_rpn_cls_fpn2": "0.000233", "loss_rpn_cls_fpn3": "0.000001", "loss_rpn_cls_fpn4": "0.000000", "loss_rpn_cls_fpn5": "0.000000", "loss_rpn_cls_fpn6": "0.000000", "lr": "0.000025", "mb_qsize": 0, "mem": 2744, "time": "0.218661"}
INFO net.py: 143: Saving parameters and momentum to /tmp/detectron-output/train/gj_2019_train/generalized_rcnn/model_iter59999.pkl
INFO net.py: 143: Saving parameters and momentum to /tmp/detectron-output/train/gj_2019_train/generalized_rcnn/model_final.pkl
INFO loader.py: 126: Stopping enqueue thread
INFO loader.py: 113: Stopping mini-batch loading thread
INFO loader.py: 113: Stopping mini-batch loading thread
INFO loader.py: 113: Stopping mini-batch loading thread
INFO loader.py: 113: Stopping mini-batch loading thread
Traceback (most recent call last):
  File "tools/train_net.py", line 132, in <module>
    main()
  File "tools/train_net.py", line 117, in main
    test_model(checkpoints['final'], args.multi_gpu_testing, args.opts)
  File "tools/train_net.py", line 127, in test_model
    check_expected_results=True,
  File "/home/qiang/detectron/detectron/core/test_engine.py", line 127, in run_inference
    all_results = result_getter()
  File "/home/qiang/detectron/detectron/core/test_engine.py", line 107, in result_getter
    multi_gpu=multi_gpu_testing
  File "/home/qiang/detectron/detectron/core/test_engine.py", line 148, in test_net_on_dataset
    dataset = JsonDataset(dataset_name)
  File "/home/qiang/detectron/detectron/datasets/json_dataset.py", line 60, in __init__
    'Ann fn \'{}\' not found'.format(dataset_catalog.get_ann_fn(name))
AssertionError: Ann fn '/home/qiang/detectron/detectron/datasets/data/gj/annotations/gj_instances.json' not found

```
**训练是完全OK！至于后面的错误，在后面已经得到解决！此处的错误你可以不用那么紧张！一切都会慢慢的好起来的！**

##5. infer_simple.py推断测试
```
qiang@qiang:~/detectron$ python tools/infer_simple.py \
   --cfg configs/gj_1gpu_e2e_faster_rcnn_R-50-FPN.yaml \
   --output-dir ~/detectron/detectron-visualizations \
   --image-ext jpg \
   --wts ~/detectron/models/model_final.pkl \
   demo

qiang@qiang:~/detectron$ python tools/infer_simple.py \
   --cfg configs/gj_1gpu_e2e_faster_rcnn_R-50-FPN.yaml \
   --output-dir ~/detectron/detectron-visualizations \
   --image-ext jpg \
   --wts ~/detectron/models/model_final.pkl \
   /home/qiang/detectron/detectron/datasets/data/gj/test_dataset
```
![图片标题](https://leanote.com/api/file/getImage?fileId=5c78be7bab64410dee001ce2)
```
#dummy_coco_dataset = dummy_datasets.get_coco_dataset() #coco数据classes
#dummy_coco_dataset = dummy_datasets.get_voc_dataset() #voc数据classes
dummy_coco_dataset = dummy_datasets.get_gj_dataset() #gj数据classes  
```
![图片标题](https://leanote.com/api/file/getImage?fileId=5c78bea1ab64410dee001ceb)

命令解释：
```
qiang@qiang:~/detectron$ python tools/infer_simple.py \
   --cfg configs/gj_1gpu_e2e_faster_rcnn_R-50-FPN.yaml \         #指定配置文件
   --output-dir ~/detectron/detectron-visualizations \           #测试结果保存路径
   --image-ext jpg \          #指定图片类型，这里是jpg，也可以是其他如png等
   --wts ~/detectron/models/model_final.pkl \                      #指定训练模型文件
   /home/qiang/detectron/detectron/datasets/data/gj/test_dataset   #指定测试的图片路径！
```


##6. 生成测试json文件（image_2_json.py）
[参考来源](https://blog.csdn.net/Mr_health/article/details/80817934)

``` 
"""
生成测试用的json文件，对测试数据集生成coco格式
没有annotation
只有图片的信息和categories的信息
"""
# -*- coding: utf-8 -*-
# @Author: liuqiang
# @Date:   2019-02-27 18:10:53

import os
import json
import collections
import cv2 
 
coco = collections.OrderedDict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []
 
image_id = 000001     #对每张图片进行编号，初始编号
category_item_id = 1
classname = ['gj']    #更改为你自己的类别,可以仿照COCO或voc的类别
 
 
def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):   #files是文件夹下所有文件的名称
    for filespath in files:            #依次取文件名
      filepath = os.path.join(root, filespath)    #构成绝对路径
      extension = os.path.splitext(filepath)[1][1:]  #os.path.splitext(path)  #分割路径，返回路径名和文件后缀 其中[1]为后缀.png，再取[1:]得到png
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles   #返回dir中所有文件的绝对路径
 
 
def addCatItem(name):
    '''
    增加json格式中的categories部分
    '''
    global category_item_id
    category_item = collections.OrderedDict()
    category_item['supercategory'] = 'none'
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_item_id += 1
 
def addImgItem(img_name, size):
    '''
    增加json格式中的images部分
    '''
    global image_id
    if img_name is None:
        raise Exception('Could not find Picture file.')
    if size['width'] is None:
        raise Exception('Could not find width.')
    if size['height'] is None:
        raise Exception('Could not find height.')
    #image_item = dict()    #按照一定的顺序，这里采用collections.OrderedDict()
    image_item = collections.OrderedDict()
    image_item['file_name'] = img_name  
    image_item['width'] = size['width']   
    image_item['height'] = size['height']
    image_item['id'] = image_id
    coco['images'].append(image_item) 
    image_id = image_id+1
 
 
def WriteCOCOFiles(pic_path):
    for idx,path in enumerate(pic_path):
        size = {}
        imgname = os.path.basename(path) #得到除去后缀的名字
        img = cv2.imread(path)
        size['height'] = img.shape[0] 
        size['width'] = img.shape[1]  
        addImgItem(imgname, size)
        print('add image with {} and {}'.format(imgname, size))
    for idx, obj in enumerate(classname):
        addCatItem(obj)          
 
 
if __name__ == '__main__':
    pic_dir = './test_dataset'              #图片存放的路径
    json_file = './gj_instances_test.json'  #生成的coco路径
    pic_path = GetFileFromThisRootDir(pic_dir,ext = None)  #每一个图片的路径
    WriteCOCOFiles(pic_path)
    json.dump(coco, open(json_file, 'w'))
 
```
![图片标题](https://leanote.com/api/file/getImage?fileId=5c78c870ab64411022001ea9)

##7. test_net.py测试
```
python tools/test_net.py \
    --cfg configs/gj_1gpu_e2e_faster_rcnn_R-50-FPN.yaml \
    TEST.WEIGHTS ~/detectron/models/model_final.pkl \
    NUM_GPUS 1
```
测试返回
```

......限于篇幅，前面省略......

INFO net.py: 133: res5_2_branch2a_b preserved in workspace (unused)
INFO net.py: 133: res5_2_branch2b_b preserved in workspace (unused)
INFO net.py: 133: res5_2_branch2c_b preserved in workspace (unused)
[I net_dag_utils.cc:102] Operator graph pruning prior to chain compute took: 5.7737e-05 secs
[I net_dag_utils.cc:102] Operator graph pruning prior to chain compute took: 5.1351e-05 secs
[I net_async_base.h:211] Using specified CPU pool size: 4; device id: -1
[I net_async_base.h:216] Created new CPU pool, size: 4; device id: -1
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 1/200 0.388s + 0.001s (eta: 0:01:17)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 11/200 0.135s + 0.000s (eta: 0:00:25)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 21/200 0.123s + 0.000s (eta: 0:00:22)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 31/200 0.119s + 0.000s (eta: 0:00:20)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 41/200 0.116s + 0.000s (eta: 0:00:18)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 51/200 0.115s + 0.000s (eta: 0:00:17)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 61/200 0.114s + 0.000s (eta: 0:00:15)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 71/200 0.114s + 0.000s (eta: 0:00:14)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 81/200 0.113s + 0.000s (eta: 0:00:13)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 91/200 0.113s + 0.000s (eta: 0:00:12)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 101/200 0.112s + 0.000s (eta: 0:00:11)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 111/200 0.112s + 0.000s (eta: 0:00:10)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 121/200 0.112s + 0.000s (eta: 0:00:08)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 131/200 0.112s + 0.000s (eta: 0:00:07)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 141/200 0.112s + 0.000s (eta: 0:00:06)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 151/200 0.112s + 0.000s (eta: 0:00:05)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 161/200 0.112s + 0.000s (eta: 0:00:04)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 171/200 0.112s + 0.000s (eta: 0:00:03)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 181/200 0.111s + 0.000s (eta: 0:00:02)
INFO test_engine.py: 286: im_detect: range [1, 200] of 200: 191/200 0.111s + 0.000s (eta: 0:00:01)
INFO test_engine.py: 319: Wrote detections to: /home/qiang/detectron/test/gj_2019_test/generalized_rcnn/detections.pkl
INFO test_engine.py: 161: Total inference time: 43.409s
INFO task_evaluation.py:  76: Evaluating detections
Traceback (most recent call last):
  File "tools/test_net.py", line 116, in <module>
    check_expected_results=True,
  File "/home/qiang/detectron/detectron/core/test_engine.py", line 127, in run_inference
    all_results = result_getter()
  File "/home/qiang/detectron/detectron/core/test_engine.py", line 107, in result_getter
    multi_gpu=multi_gpu_testing
  File "/home/qiang/detectron/detectron/core/test_engine.py", line 163, in test_net_on_dataset
    dataset, all_boxes, all_segms, all_keyps, output_dir
  File "/home/qiang/detectron/detectron/datasets/task_evaluation.py", line 60, in evaluate_all
    dataset, all_boxes, output_dir, use_matlab=use_matlab
  File "/home/qiang/detectron/detectron/datasets/task_evaluation.py", line 98, in evaluate_boxes
    'No evaluator for dataset: {}'.format(dataset.name)
NotImplementedError: No evaluator for dataset: gj_2019_test
```
![图片标题](https://leanote.com/api/file/getImage?fileId=5c78f7d3ab64410dee0028b5)
最后可能会报错说缺少gj_2019_test，但其实是没有影响的，主要是因为我们前面生成的json文件中包含了category（类别）这个部分，因为我也尝试过json文件中不生成category，这时候运行测试命令则没有报错。
通过下图我们也可以看到，报错的位置在task_evaluation.py。但是尽管报错了，我们所需要的检测结果detection.pkl还是会生成的。
既然这样的话为什么还要生成catogory呢？因为我成生成detection.pkl后总需要可视化的呀，也要看看检测的结果噻。也就是需要运行tools/visualize_results.py，该文件需要json文件中catogary部分的支持。

##8. visualize_results.py可视化测试
```
python tools/visualize_results.py \
    --dataset gj_2019_test \
    --detections test/detections.pkl \
    --output-dir test

python tools/visualize_results.py --dataset gj_2019_test --detections test/detections.pkl --output-dir test
```
**注意：gj_2019_test是直接用的在dataset_catalog.py文件中取得名字！！**

命令返回
```
qiang@qiang:~/detectron$ python tools/visualize_results.py \
>     --dataset gj_2019_test \
>     --detections test/detections.pkl \
>     --output-dir test
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
1/200
11/200
21/200
31/200
41/200
51/200
61/200
71/200
81/200
91/200
101/200
111/200
121/200
131/200
141/200
151/200
161/200
171/200
181/200
191/200
```
![图片标题](https://leanote.com/api/file/getImage?fileId=5c78f9f2ab64410dee002940)

**指定可视化输出格式为jpg**
/home/qiang/detectron/detectron/utils/vis.py **参数默认是ext='pdf' 修改为 ext='jpg'**
![图片标题](https://leanote.com/api/file/getImage?fileId=5c7cbdf5ab64414599005af4)

**第2次训练开始时间：2019.03.02 12.19 48**

## 9. 生成提交的CSV文件
（1）为了输出提交的CSV文件，修改了/home/bobo/liuq/detectron/tools/visualize_results.py
```
def vis(dataset, detections_pkl, thresh, output_dir, limit=0):
    ds = JsonDataset(dataset)
    roidb = ds.get_roidb()

    dets = load_object(detections_pkl)

    assert all(k in dets for k in ['all_boxes', 'all_segms', 'all_keyps']), \
        'Expected detections pkl file in the format used by test_engine.py'

    all_boxes = dets['all_boxes']
    all_segms = dets['all_segms']
    all_keyps = dets['all_keyps']

    def id_or_index(ix, val):
        if len(val) == 0:
            return val
        else:
            return val[ix]
    box_locations = []
    img_ids = []
    for ix, entry in enumerate(roidb):
        if limit > 0 and ix >= limit:
            break
        if ix % 10 == 0:
            print('{:d}/{:d}'.format(ix + 1, len(roidb)))

        im = cv2.imread(entry['image'])
        im_name = os.path.splitext(os.path.basename(entry['image']))[0]

        cls_boxes_i = [
            id_or_index(ix, cls_k_boxes) for cls_k_boxes in all_boxes
        ]
        cls_segms_i = [
            id_or_index(ix, cls_k_segms) for cls_k_segms in all_segms
        ]
        cls_keyps_i = [
            id_or_index(ix, cls_k_keyps) for cls_k_keyps in all_keyps
        ]

        boxes = vis_utils.vis_one_image(  #2019.03.05，添加返回了boxes数据
            im[:, :, ::-1],
            '{:d}_{:s}'.format(ix, im_name),
            os.path.join(output_dir, 'vis'),
            cls_boxes_i,
            segms=cls_segms_i,
            keypoints=cls_keyps_i,
            thresh=thresh,
            box_alpha=0.8,
            dataset=ds,
            show_class=True
        )
        #----------------------------------------------------------------------------------- 
        #2019.03.05添加，四川大学图像信息研究所614室 
        boxes = list(map(lambda x:str(x[0]) + ' ' + str(x[1]) + ' ' + str(x[2]) + ' ' + str(x[3]), boxes.tolist()))
        box_locations.extend(boxes)
        img_ids.extend([im_name+'.jpg'] * len(boxes))
    return box_locations, img_ids
        #----------------------------------------------------------------------------------- 

if __name__ == '__main__':
    import pandas as pd
    opts = parse_args()
    df = pd.DataFrame()
    bboxes, ids = vis(    #2019.03.05，添加返回bboxes, ids数据
        opts.dataset,
        opts.detections,
        opts.thresh,
        opts.output_dir,
        limit=opts.first
    )
    #----------------------------------------------------------------------------------- 
    #2019.03.05添加，四川大学图像信息研究所614室   
    df["id"] = ids
    df["pos"] = bboxes
    df.to_csv("/home/bobo/liuq/detectron/test/submit/submit.csv", index=False, header=None)
    #-----------------------------------------------------------------------------------     
```
（2）为了输出提交的CSV文件，修改了/home/bobo/liuq/detectron/detectron/utils/vis.py
```
def vis_one_image(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='jpg', out_when_no_box=False):
    """Visual debugging of detections."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if (boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh) and not out_when_no_box:
        return

    dataset_keypoints, _ = keypoint_utils.get_keypoints()

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    if boxes is None:
        sorted_inds = [] # avoid crash when 'boxes' is None
    else:
        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        # show box (off by default)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='g',
                          linewidth=0.5, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, dataset),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    plt.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

                if kps[2, i2] > kp_thresh:
                    plt.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)
    #----------------------------------------------------------------------------------- 
    #2019.03.05添加，四川大学图像信息研究所614室
    # result_boxes = boxes[:, :4].astype(int)
    return boxes[:, :4].astype(int)
    #----------------------------------------------------------------------------------- 
    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    plt.close('all')
```



 

