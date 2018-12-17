import os, csv
from time import gmtime, strftime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random, math
import tensorflow as tf
import configs.config as cfg
import networks.mobilenet_V2 as model
import data.datapipe as datasets
import scipy.misc as sm

def inference():

    os.environ["CUDA_VISIBLE_DEVICES"] ="1"

    # base_dir = 'data/OIDv4/validation'
    base_dir = 'data/images'
    filenames = sorted(os.listdir(base_dir))
    print(len(filenames), filenames[0])

    classes=[]
    with open('data/train_class_names.csv', 'r', newline='') as f:
        lines = csv.reader(f)
        for i in lines:
            id = i[0]
            c_name = i[-1]
            classes.append(c_name)
        f.close()
    images_list = []
    for i in range(len(filenames)):
        img = sm.imread(base_dir+'/'+filenames[i], mode='RGB')
        images_list.append(img)

    # Create global_step
    with tf.device('/cpu:0'):
        inputs = tf.placeholder(dtype=tf.float32, shape=(None, None, 3), name='input')
        processing_X = datasets._preprocess_for_test(inputs)
        processing_X = tf.expand_dims(processing_X, 0)
        print(processing_X.shape)

    with tf.device('/cpu:0'):
        nets = model.MobileNet_V2(processing_X,
                                num_classes=cfg.FLAGS.num_classes,
                                is_training=False,
                                weight_decay=cfg.FLAGS.weight_decay)

        probs = nets.end_points['probs']
        probs = tf.cast(probs, tf.float32)
        print(probs)

        """ create saver and initialize variables """
        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        ckpt_state = tf.train.get_checkpoint_state('output/MB_V2')
        if ckpt_state is not None:
            model_path = tf.train.latest_checkpoint('output/MB_V2')
        else:
            model_path = 'output/models/MB_V2_ML_Recognition_final.ckpt'

        print('model_path : ', model_path)

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(init_op)
            saver.restore(sess, model_path)
            print ('Evaluating...')
            for i in range(len(images_list)):
                im = images_list[i]
                print(filenames[i], im.shape)
                preds = sess.run(probs, feed_dict={inputs: im})
                inds = np.where(preds>0.5)[1]

                result = {}
                for ii in range(len(inds)):
                    c_id = inds[ii]
                    score = preds[0][c_id]
                    result[score] = c_id

                pred_text = ""
                for k, v in sorted(result.items(), key=lambda t:t[0], reverse=True):
                    score = k
                    class_name = classes[v]
                    pred_text = pred_text + class_name + "(" + str(score) + ")\n"

                r_img = Image.fromarray(im)
                d = ImageDraw.Draw(r_img)
                font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
                d.text((10, 10), pred_text, fill=(255, 0, 0), font=font)
                r_img.save(os.path.join('data/result', filenames[i]))

            sess.close()


if __name__ == "__main__":
    inference()
