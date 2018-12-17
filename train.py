import os
from time import gmtime, strftime
import numpy as np
from PIL import Image
import scipy.misc
import random
import tensorflow as tf
import configs.config as cfg
import networks.mobilenet_V2 as model
import data.datapipe as datasets
import scipy.misc as sm
import tensorflow.contrib.slim as slim

def _get_learning_rate(num_sample_per_epoch, global_step):
    decay_step = int((num_sample_per_epoch / cfg.FLAGS.batch_size) * cfg.FLAGS.num_epochs_per_decay)
    return tf.train.exponential_decay(cfg.FLAGS.learning_rate,
                                      global_step,
                                      decay_step,
                                      cfg.FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')

def train():
    """ Only allocate to visible gpu memory """
    os.environ["CUDA_VISIBLE_DEVICES"] ="0"

    dataset_dir = cfg.FLAGS.dataset_dir
    surfix = cfg.FLAGS.surfix
    num_classes = cfg.FLAGS.num_classes
    batch_size = cfg.FLAGS.batch_size
    num_images = 1743042

    # Create global_step
    with tf.device('/cpu:0'):
        global_step = tf.train.create_global_step()
        """ load data """
        target_labels = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_classes), name='target_labels')
        input_image = tf.placeholder(dtype=tf.float32, shape=(batch_size, 512, 512, 3),  name='input_images')
        img_list=[]
        for i in range(batch_size):
            img_list.append(datasets._preprocess_for_training(input_image[i]))

        image = tf.stack(img_list, axis=0, name='target_images')

    with tf.device('/gpu:0'):
        nets = model.MobileNet_V2(image,
                                num_classes=num_classes,
                                is_training=True,
                                weight_decay=cfg.FLAGS.weight_decay)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logits = nets.end_points['logits']

        """ compute total loss """
        all_losses = []
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_labels,
                                                                logits=logits,
                                                                name='sigmoidCE')
        # cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=target_labels,
        #                                                          logits=logits,
        #                                                          pos_weight=12,
        #                                                          name='sigmoidCE')


        cross_loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1), name='cross_loss')
        all_losses.append(cross_loss)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularization_loss = tf.add_n(regularization_losses, name='sum_regularization_loss')
        all_losses.append(regularization_loss)
        total_loss = tf.add_n(all_losses)

        """ Configure the optimization procedure. """
    with tf.device('/gpu:0'):
        learning_rate = _get_learning_rate(num_images, global_step)
    with tf.device('/gpu:0'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                              decay=cfg.FLAGS.rmsprop_decay,
                                              momentum=cfg.FLAGS.rmsprop_momentum,
                                              epsilon=cfg.FLAGS.opt_epsilon)

        """ Variables to train """
        train_vars = tf.trainable_variables()
        grad_op = optimizer.minimize(total_loss, global_step=global_step, var_list=train_vars)
        update_ops.append(grad_op)
        update_op = tf.group(*update_ops)

        """ Estimate accrancy """
        pred_cls = tf.round(nets.end_points['probs'])
        TP = tf.reduce_sum(pred_cls * target_labels)
        FP = tf.reduce_sum(pred_cls * (1-target_labels))
        accurancy = TP / (TP + FP + 1e-10)

        """ set Summary and log info """
    with tf.device('/cpu:0'):
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        """ Add summaries for total loss """
        summaries.add(tf.summary.scalar('losses/total_loss', total_loss))
        summaries.add(tf.summary.scalar('losses/regularization_loss', regularization_loss))
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        logdir = os.path.join(cfg.FLAGS.train_dir, strftime('MB_V2/%Y%m%d%H%M%S', gmtime()))
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        summary_writer = tf.summary.FileWriter(logdir, graph=tf.Session().graph)

        """ create saver and initialize variables """
        saver = tf.train.Saver(max_to_keep=3)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
    with tf.device('/gpu:0'):
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            sess.run(init_op)
            ckpt_statue = tf.train.get_checkpoint_state('output/MB_V2')
            if ckpt_statue:
                lastest_ckpt = tf.train.latest_checkpoint('output/MB_V2')
                print(lastest_ckpt)
                saver.restore(sess, lastest_ckpt)
            else:
                print('no+++++++++++++++++')
                # variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MobilenetV2')
                # for i in variables_to_restore:
                #     print(i)
                # re_saver = tf.train.Saver(var_list=variables_to_restore)
                # re_saver.restore(sess, 'output/pretrained_models/mobilenet_v2_1.0_224.ckpt')
            try:
                print("========== Start training... =============")
                b=0
                image_index = -1
                image_files, gt_labels = datasets.get_dataset(surfix, dataset_dir, batch_size)
                image_ids = list(range(len(image_files)))

                while True:
                    image_index = (image_index + 1) % len(image_files)
                    # shuffle images if at the start of an epoch
                    if image_index == 0:
                        np.random.shuffle(image_ids)

                    if b == 0:
                        batch_images = np.zeros((batch_size, 512, 512, 3), dtype=np.int32)
                        batch_gt_labels = np.zeros((batch_size, num_classes), dtype=np.float32)

                    # load image and gt_label for the image.
                    image_id = image_ids[image_index]
                    im = sm.imread(image_files[image_id], mode='RGB')
                    batch_images[b] = sm.imresize(im, (512, 512))

                    onehot = np.zeros(shape=(num_classes), dtype=np.float32)
                    onehot[gt_labels[image_id]] = 1.0
                    batch_gt_labels[b] = onehot
                    # print(batch_images[b].shape, batch_gt_labels[b], image_files[image_id], image_id)
                    b+=1

                    if b >= batch_size:
                        feed_dict = {input_image : batch_images, target_labels : batch_gt_labels}
                        _, current_step, losses, tar_labels, pred_labels, acc = sess.run(
                            [update_op, global_step, total_loss, target_labels, pred_cls, accurancy],
                            feed_dict=feed_dict)

                        num_epoch = (current_step*batch_size) / num_images

                        print(""" iter %d / %d: total_loss %.4f acc : %.4f """ % (current_step, num_epoch, losses, acc))

                        """ write summary """
                        if current_step % 1000 == 0:
                            """ write summary """
                            summary = sess.run(summary_op, feed_dict=feed_dict)
                            summary_writer.add_summary(summary, current_step)

                            tp = np.sum(pred_labels * tar_labels)
                            fp = np.sum(pred_labels * (1 - tar_labels))

                            acc = tp / (tp + fp + 1e-10)
                            print(tp, fp, acc)


                        """ save trained model datas """
                        if current_step % 3000 == 0:
                            saver.save(sess, "output/MB_V2/Scene_recognition.ckpt", global_step=current_step)

                        if current_step == cfg.FLAGS.max_iters:
                            print('step is reached the maximum iteration')
                            print('Done training!!!!!')
                        b = 0

            except tf.errors.OutOfRangeError:
                print('Error is occured and stop the training')
            finally:
                saver.save(sess, './output/models/MB_V2_Scene_recognition_final_ckpt')
            sess.close()

        print('Done.')

if __name__ == "__main__":
    train()