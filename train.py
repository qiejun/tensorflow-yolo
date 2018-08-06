from yolo.yolo_net import *
from util.voc import *
from yolo.config import batch_size, epochs, voc_path


def main():
    yolo = YOLO()
    merge = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input_x = yolo.input_x
        input_y = yolo.input_y
        traing_bn = yolo.traing_bn
        loss = yolo.loss
        opt = yolo.opt
        iters = 0
        total_loss = 0
        saver = tf.train.Saver(max_to_keep=10)
        writer = tf.summary.FileWriter('./log', sess.graph)
        for e in range(epochs):
            data = load_batch(voc_path, batch_size)
            for imgs, labels in data:
                _, loss_ = sess.run([opt, loss], feed_dict={input_x: imgs, input_y: labels, traing_bn: True})
                total_loss = total_loss + loss_
                if iters % 10 == 0 and iters > 0:
                    train_summary = sess.run(merge, feed_dict={input_x: imgs, input_y: labels, traing_bn: True})
                    writer.add_summary(train_summary, iters)
                    print('epoch:', e, ',iteration:', iters, ',loss:', total_loss / 10)
                    total_loss = 0
                iters = iters + 1
            if e == 20:
                yolo.learning_rate = 1e-4
            if e >= 10 and e % 10 == 0:
                saver.save(sess, './save/save.ckpt' + str(e))


if __name__ == '__main__':
    main()
