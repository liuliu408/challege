"""
通过训练数据和train_label目录的文件生成训练xml数据
每幅图对应一个xml文档！！
文档名：直接用训练图片名命令～～
在执行完：python step1A_CSV_2_train_label.py 后执行！

重要提示：务必在当前目录下先创建 train_xml 文件夹！！
"""

import csv
import os,sys
from glob import glob
from PIL import Image
 
src_img_dir = r'./train_dataset'
src_txt_dir = r'./train_label'
src_xml_dir = r'./train_xml'
 
img_lists = glob(src_img_dir + '/*jpg')
img_basenames = []
for item in img_lists:
    img_basenames.append(os.path.basename(item))
 
img_names = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)
 
 
for img in img_names:
    im = Image.open((src_img_dir + os.sep + img + '.jpg'))     #os.sep根据你所处的平台，自动采用相应的分隔符号。是'\' 还是'/'
    width, height = im.size
    gt = open(src_txt_dir + os.sep + img + '.txt').read().splitlines()
    xml_file = open((src_xml_dir + os.sep + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2019</folder>\n')
    xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
 
    for img_each_label in gt:
        spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str('gj') + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(spt[0]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(spt[1]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(spt[2]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(spt[3]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
 
    xml_file.write('</annotation>')
