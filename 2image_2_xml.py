
"""
生成xml文件
重要提示：务必在当前目录下先创建 test_xml 文件夹！！用于保存生成的xml文件
"""

# -*- coding: utf-8 -*-
# @Author: liuqiang
# @Date:   2019-02-27 18:10:53

import csv
import os,sys
from glob import glob
from PIL import Image
 
src_img_dir = r'./test_dataset'
src_xml_dir = r'./test_xml'
 
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

    xml_file = open((src_xml_dir + os.sep + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>gj2019</folder>\n')
    xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n') 
    xml_file.write('</annotation>')


