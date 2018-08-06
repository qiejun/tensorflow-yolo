classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}

class_num = {y:x for x,y in classes_num.items()}

batch_size = 8
epochs = 101
vgg16_path = './vgg16/vgg16.npy'
voc_path = 'E:/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'