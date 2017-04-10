import os
import tensorflow as tf
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt

def create_record():
    writer = tf.python_io.TFRecordWriter("getpostdata.tfrecords")
    data_path =  "train/color/"
    label_path = "train/after_mask/"
    for img_num in range(20000):
        color_path = data_path + str(img_num) + ".png"
        normal_path = label_path + str(img_num) + ".png"
        img = Image.open(color_path)
        img = img.resize((128, 128))
        img_raw = img.tobytes() 
        label = Image.open(normal_path)
        label = label.resize((128,128))
        label_raw = label.tobytes()
        example = tf.train.Example(features=tf.train.Features(
            feature={
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),          

 }))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                          
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128,3])
    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [128, 128,3])
    return img,label

if __name__ == '__main__':
    if not os.path.exists("getpostdata.tfrecords"):
        create_record()
    img, label = read_and_decode("getpostdata.tfrecords")

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    print(img.shape,label.shape)
    print(img_batch.shape,label_batch.shape)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val, l= sess.run([img_batch, label_batch])
            count=0
            count +=np.count_nonzero(l)
        if not os.path.exists("output"):
            os.makedirs('output')
        print(np.sum(l*l,axis=3))
        print(np.shape(l))
        pic = Image.fromarray(l[0,:,:,:],'RGB')
        pic.save('output/my.png')
        pic.show()
        pic2 = Image.fromarray(val[0,:,:,:],'RGB')
        pic2.save('output/my1.png')
        pic2.show()
           
