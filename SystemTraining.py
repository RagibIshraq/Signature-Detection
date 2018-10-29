import Preprocessing
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


batch_size = 32

#input your signature class names here. That means the folder names from test_own_data.

classes = []

#Example is given bellow.


# classes = ['14_07','14_08','14_09','14_11','14_12','14_20','14_22','14_24','14_26','14_27','14_29','14_32',
#            '14_36','14_37','14_41','14_43','14_44','14_48','14_49','14_50','14_53','14_55','14_56','14_57',
#            '14_60','14_61','14_63','14_64','14_65','14_66','14_69','14_70','14_71','14_72','14_139','14_333',
#            '14_420',
#            '15_01','15_02','15_03','15_06','15_09','15_11','15_12','15_15','15_16','15_17','15_22','15_23',
#            '15_24','15_27','15_30','15_32','15_34','15_35','15_36','15_37','15_39','15_42','15_43','15_44',
#            '15_46','15_47','15_52','15_56','15_58','15_59','15_61','15_62','15_63','15_64','15_75','15_83',
#            '15_222',
#            '16_02','16_06','16_09','16_11','16_12','16_15','16_17','16_18','16_19','16_21','16_24','16_27',
#            '16_29','16_30','16_34','16_36','16_37','16_38','16_45','16_47','16_52','16_53','16_55','16_58',
#            '16_59','16_62','16_63','16_64','16_71','16_74','16_75','16_76','16_78','16_82','16_85','16_89',
#            '16_90','16_92','16_93','16_94','16_96','16_98','16_100','16_103','16_104','16_105','16_106',
#            '16_107','16_252','16_949',
#            '17_05','17_06','17_08','17_09','17_10','17_11','17_12','17_13','17_14','17_15','17_17','17_19',
#            '17_20','17_21','17_22','17_23','17_24','17_25','17_26','17_27','17_28','17_29','17_30','17_32',
#            '17_34','17_36','17_38','17_39','17_40','17_41','17_43','17_44','17_45','17_46','17_48','17_49',
#            '17_50','17_51','17_53','17_54','17_55','17_56','17_57','17_58','17_59','17_60','17_61','17_62',
#            '17_63','17_64','17_66','17_69','17_72','17_74','17_75','17_76','17_77','17_78','17_80','17_81',
#            '17_84','17_85','17_86','17_87','17_88','17_89','17_90','17_91','17_93','17_94','17_95','17_96',
#            '17_97','17_98','17_102','17_104','17_105','17_106','17_108','17_120']
num_classes = len(classes)

train_accuracy=[]
validation_accuracy=[]
validation_loss=[]
epoc_list=[]

validation_size = 0.2
img_size = 128
num_channels = 3
train_path='train_own_data'


data = Preprocessing.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("data readed seccessfully")
print("Training-set:\t\t{}".format(len(data.train.labels)))
print("Validation-set:\t{}".format(len(data.valid.labels)))



session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')


y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
#
# filter_size_conv4 = 3
# num_filters_conv4 = 64

fc_layer_size = 128

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)


    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases


    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    

    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)

# layer_conv4= create_convolutional_layer(input=layer_conv3,
#                num_input_channels=num_filters_conv3,
#                conv_filter_size=filter_size_conv4,
#                num_filters=num_filters_conv4)
          
layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
	
	acc = session.run(accuracy, feed_dict=feed_dict_train)
	val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
	msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
	print (msg.format (epoch + 1, acc, val_acc, val_loss) )

total_iterations = 0

saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations
    print ('Hello: ' +os.getcwd())
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, os.getcwd()+'\\AI-signature-verification')
    total_iterations += num_iteration
train(num_iteration=4000)




