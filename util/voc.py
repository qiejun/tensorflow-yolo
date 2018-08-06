import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from yolo.config import classes_num


def parse_single_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    filename = root.find('filename').text
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    objs = root.findall('object')
    labels = []
    for obj in objs:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        class_num = classes_num[name]
        cx = (xmax + xmin) * 0.5 / width
        cy = (ymax + ymin) * 0.5 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        labels.append([cx, cy, w, h, class_num])
    return filename, labels


def process_label(labels):
    init = np.zeros((7, 7, 2, 25))
    for label in labels:
        x, y = int(label[0] * 7), int(label[1] * 7)
        init[x, y, :, label[4]] = 1  # class num
        init[x, y, :, 20] = 1  # confidence
        init[x, y, :, -4:] = label[0:4]  # location
    return init


def load_batch(root_path, batch_size):
    xmls_path = os.path.join(root_path, 'Annotations')
    xmls = os.listdir(xmls_path)
    batch_num = len(xmls) // batch_size
    for i in range(batch_num):
        labels = []
        imgs = []
        data = xmls[batch_size * i:batch_size * (i + 1)]
        for xml_name in data:
            xml_path = os.path.join(xmls_path, xml_name)
            filename, label = parse_single_xml(xml_path)
            label = process_label(label)
            img_path = os.path.join(root_path, 'JPEGImages', filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (448, 448)) / 255
            labels.append(label)
            imgs.append(img)
        yield np.array(imgs), np.array(labels)


def bndbox_test(img, label):
    for i in range(7):
        for j in range(7):
            if label[i, j, 0, 20] == 1:
                cx, cy, w, h = label[i, j, 0, -4:] * 448
                xmin = int(cx - 0.5 * w)
                ymin = int(cy - 0.5 * h)
                cv2.rectangle(img, (xmin, ymin), (xmin + int(w), ymin + int(h)), (0, 255, 0))
    cv2.imshow('1', img)
    cv2.waitKey(0)

# data = load_batch('E:\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007',10)
#
# for imgs, labels in data:
#     img = imgs[0,...]
#     label = labels[0,...]
#     bndbox_test(img,label)
