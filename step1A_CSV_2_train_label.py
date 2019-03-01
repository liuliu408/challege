"""
智能盘点—钢筋数量AI识别
 
通过训练数据和CSV标注文件在train_label目录生成训练txt（数据标签来源与CSV文件！！）
每幅图对应一个TXT文档！！
文档名：直接用训练图片名命令～～
重要提示：务必在当前目录下先创建 train_label 文件夹！！
"""

import csv
import os,sys
from glob import glob
from PIL import Image
 
src_img_dir = r'./train_dataset'
src_txt_dir = r'./train_label'
src_xml_dir = r'./train_label_xml'
 
img_lists = glob(src_img_dir + '/*jpg')
# 返回的文件名只包括当前目录里的文件名，不包括子文件夹里的文件。字符串可以为绝对路径也可以为相对路径!
img_basenames = []
for item in img_lists:
    img_basenames.append(os.path.basename(item))
    #os.path.basename(),返回path最后的文件名
img_names = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)
 
c = []
filename = r'./train_labels.csv'
with open(filename) as f:
    reader = csv.reader(f)
    head_now = next(reader)
    l = []
    b = []
    for cow in reader:
        label = cow[0]
        l.append(label)
        bbox = cow[1]
        b.append(bbox)
label = []
for item in l:
    temp1, temp2 = os.path.splitext(item)
    label.append(temp1)
 
for img in img_names:
    img_file = src_txt_dir + os.sep + img +'.txt'
    fp = open(img_file, 'w')
    for i in range(len(label)):
        if label[i] == img:
            fp.write(str(b[i]))
            fp.write('\n')

