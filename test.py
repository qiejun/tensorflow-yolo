from yolo.yolo_net import *
import cv2
from yolo.config import class_num


class Detector():
    def __init__(self, single_img_path):
        self.img_path = single_img_path
        self.session = tf.InteractiveSession()
        self.yolo = YOLO()
        self.input_x = self.yolo.input_x
        self.logits = self.yolo.logits
        self.score, self.clas, self.locs = self.pred_process()
        self.score_run, self.clas_run, self.locs_run = self.run_tensor()

    def pred_process(self):
        pred = tf.reshape(self.logits, [-1, 7, 7, 2, 25])
        cla = pred[:, :, :, :, :20]  # [-1,7,7,2,20]
        conf = pred[:, :, :, :, 20]  # [-1,7,7,2]
        loc = pred[..., 21:]  # [-1,7,7,2,4]

        conf = tf.expand_dims(conf, axis=-1)
        scores = cla * conf  # [-1,7,7,2,20]
        scores = tf.clip_by_value(scores,0.,1.)
        cla_idx = tf.argmax(scores, -1)  # [-1,7,7,2]
        scores_max = tf.reduce_max(scores, axis=-1)  # [-1,7,7,2]

        mask = scores_max >= 0.5  # [-1,7,7,2]
        scores = tf.boolean_mask(scores_max, mask)  # [?,1]
        clas = tf.boolean_mask(cla_idx, mask)  # [?,1]
        locs = tf.boolean_mask(loc, mask)  # [?,4]
        locs = tf.nn.relu(locs)
        return scores, clas, locs

    def run_tensor(self):
        img = cv2.imread(self.img_path)
        img = np.array(cv2.resize(img, (448, 448))) / 255
        img = np.expand_dims(img, axis=0)
        saver = tf.train.Saver()
        saver.restore(self.session, './save/save.ckpt100')
        score, clas, locs = self.session.run([self.score, self.clas, self.locs],
                                             feed_dict={self.input_x: img, self.yolo.traing_bn: False})
        print(clas)
        return score, clas, locs

    def non_max(self, score, clas, locs):
        non_score = []
        non_clas = []
        non_locs = []
        iou_cache = []
        if len(score) != 0:
            non_score.append(
                score[np.argmax(score)]
            )
            non_clas.append(
                clas[np.argmax(score)]
            )
            non_locs.append(
                locs[np.argmax(score)]
            )
            for index in reversed(np.argsort(score)):
                if clas[index] not in non_clas and locs[index][2] > 0 and locs[index][3] > 0:
                    non_score.append(score[index])
                    non_clas.append(clas[index])
                    non_locs.append(locs[index])
                    break
                for cla_index, cla in enumerate(non_clas):
                    if clas[index] == cla:
                        iou_ = self.iou_one(non_locs[cla_index], locs[index])
                        iou_cache.append(iou_)
                if len([iou for iou in iou_cache if iou > 0.5]) == 0:
                    non_score.append(score[index])
                    non_clas.append(clas[index])
                    non_locs.append(locs[index])
                iou_cache = []
        return non_score, non_clas, non_locs

    def img_detector(self):
        img = cv2.imread(self.img_path)
        img_height = img.shape[0]
        img_width = img.shape[1]
        score_list, clas_list, locs_list = self.non_max(self.score_run, self.clas_run, self.locs_run)
        if len(score_list) != 0:
            for index in range(len(score_list)):
                score, cla, loc = score_list[index], clas_list[index], locs_list[index]
                cx, cy, w, h = loc
                if w > 0 and h > 0:
                    xmin = int((cx - w * 0.5) * img_width)
                    ymin = int((cy - h * 0.5) * img_height)
                    xmax = int((cx + w * 0.5) * img_width)
                    ymax = int((cy + h * 0.5) * img_height)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0))
                    cv2.putText(img, class_num[cla] + ':' + str(round(score, 2)), (xmin, ymin),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
            cv2.imshow('img', img)
            cv2.waitKey(0)
        else:
            cv2.imshow('img', img)
            cv2.waitKey(0)

    def iou_one(self, box1, box2):
        xmin1 = box1[0] - box1[2] / 2.
        ymin1 = box1[1] - box1[3] / 2.
        xmax1 = box1[0] + box1[2] / 2.
        ymax1 = box1[1] + box1[3] / 2.
        xmin2 = box2[0] - box2[2] / 2.
        ymin2 = box2[1] - box2[3] / 2.
        xmax2 = box2[0] + box2[2] / 2.
        ymax2 = box2[1] + box2[3] / 2.
        interw = np.maximum(0., np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2))
        interh = np.maximum(0., np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2))
        inter = interw * interh
        union = box1[2] * box1[3] + box2[2] * box2[3] - inter + 1e-10
        return np.clip(inter / union, 0, 1)


if __name__ == '__main__':
    solver = Detector('E:\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\\001881.jpg')
    solver.img_detector()
