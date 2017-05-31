import os
import tensorflow as tf
from PIL import Image
import numpy as np

def create_train_record():
    writer = tf.python_io.TFRecordWriter("get_train_data.tfrecords")
    data_path =  "train/color/"
    label_path = "train/normal/"
    masker_path = "train/mask/"
    for img_num in range(20000):
        color_path = data_path + str(img_num) + ".png"
        normal_path = label_path + str(img_num) + ".png"
        mask_path = masker_path + str(img_num) + ".png"
        img = Image.open(color_path)
        img = img.resize((128, 128))
        img_raw = img.tobytes() 
        label = Image.open(normal_path)
        label = label.resize((128,128))
        label_raw = label.tobytes()
        mask = Image.open(mask_path)	
        mask = mask.resize((128,128))
        mask_raw = mask.tobytes()

        example = tf.train.Example(features=tf.train.Features(
            feature={
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'mask_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_raw]))
           

 }))
        writer.write(example.SerializeToString())
    writer.close()

def train_read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'mask_raw' : tf.FixedLenFeature([], tf.string),

                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128,3])
    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [128, 128,3])
    mask = tf.decode_raw(features['mask_raw'], tf.uint8)
    mask = tf.reshape(mask, [128, 128,1])
    mask = tf.transpose([mask,mask,mask])
    mask = tf.reshape(mask,[128,128,3])
    mask = tf.transpose(mask,perm=[1,0,2])
    mask = mask/255
    label = mask*label
    img_batch, label_batch= tf.train.shuffle_batch([img,label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    #print(img_batch.shape)
    return img_batch,label_batch

def create_test_record():
    writer = tf.python_io.TFRecordWriter("get_test_data.tfrecords")
    data_path =  "test/color/"
    masker_path = "test/mask/"
    for img_num in range(2000):
        color_path = data_path + str(img_num) + ".png"
        mask_path = masker_path + str(img_num) + ".png"
        img = Image.open(color_path)
        img = img.resize((128, 128))
        img_raw = img.tobytes() 
        mask = Image.open(mask_path)	
        mask = mask.resize((128,128))
        mask_raw = mask.tobytes()

        example = tf.train.Example(features=tf.train.Features(
            feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),          
            'mask_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_raw]))
 }))
        writer.write(example.SerializeToString())
    writer.close()

def test_read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw' : tf.FixedLenFeature([], tf.string),  
                                           'mask_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128,3])
    mask = tf.decode_raw(features['mask_raw'], tf.uint8)
    mask = tf.reshape(mask, [128, 128,1])
    mask = tf.transpose([mask,mask,mask])
    mask = tf.reshape(mask,[128,128,3])
    mask = tf.transpose(mask,perm=[1,0,2])
    mask = mask/255
    #test_batch, mask_batch= tf.train.shuffle_batch([img,mask],
                                                    #batch_size=30, capacity=2000,
                                                    #min_after_dequeue=1000)
    #print(test_batch.shape)
    #return test_batch, mask_batch
    print img.shape
    img2 = tf.reshape(img, [-1,128,128,3])
    print img2.shape
    return img2, mask




if __name__ == '__main__':
    if not os.path.exists("get_train_data.tfrecords"):
        create_train_record()
    img_batch, label_batch= train_read_and_decode("get_train_data.tfrecords")
    if not os.path.exists("get_test_data.tfrecords"):
        create_test_record()
    test_batch, m_batch= test_read_and_decode("get_train_data.tfrecords")
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val, l= sess.run([img_batch, label_batch])
            test_val, test_m = sess.run([test_batch, m_batch])
            print(val.shape)
            print(test_val.shape)
            count=0
            count +=np.count_nonzero(l)    
            pic = Image.fromarray(l[i,:,:,:],'RGB')
            pic.save('my.png')
            pic.show()
            pic2 = Image.fromarray(val[i,:,:,:],'RGB')
            pic2.save('my1.png')
            pic2.show()


