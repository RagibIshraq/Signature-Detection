import cv2
import os
import glob
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import os, glob, cv2
import sys, argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = sys.argv[1]
filename = dir_path + '/' + image_path + '/'
path = os.path.join('test_own_data', filename, '*g')
files = glob.glob(path)
#print(files)

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
image_size = 128
num_channels = 3

totalExperimented =0;
truePositive=0;
#decisionMartrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
for i in files:

    filename = i

    #print(i)
    p = 0
    className = ""
    chk2=0;
    i = i[::-1]
    for kk in i:
        if kk == '.':
            chk2=1;
        elif kk == '\\':
            break;
        elif kk == '/':
            break;
        elif chk2 == 1:
            className = className + kk;

    # for k in i:
    #     if i[p] == 't':
    #         if i[p+1] == 'e':
    #             if i[p+2] == 's':
    #                 if i[p+3] == 't':
    #                     q=p+5
    #                     r=0
    #                     for l in i:
    #                         className = className + i[q]
    #                         q=q+1
    #                         r=r+1
    #                         if r == 3:
    #                             break
    #
    #                 break
    #
    #     # print(i[p])
    #     # print(p)
    #     p = p + 1

    # image_path=sys.argv[1]
    # filename = dir_path +'/'+'test/bag.1.jpg'






    #className = className
    className = className[::-1]
    #print(className)

    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    ## Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('AI-signature-verification.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, num_classes))

    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    #print('Result: ')
    #print(result)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]


    # inputClass = 0
    # #print(className)
    # if className == classes[0]:
    #     inputClass = 0
    # elif className == classes[1]:
    #     inputClass = 1
    # elif className == classes[2]:
    #     inputClass = 2
    # elif className == classes[3]:
    #     inputClass = 3
    # elif className == classes[4]:
    #     inputClass = 4
    # elif className == classes[5]:
    #     inputClass = 5
    # elif className == classes[6]:
    #     inputClass = 6
    # elif className == classes[7]:
    #     inputClass = 7
    # elif className == classes[8]:
    #     inputClass = 8
    # elif className == classes[9]:
    #     inputClass = 9
    # m1 = max(result)

    #print(inputClass)
    #m = max(m1)
    j = 1;
    val =0;
    probableClass = 0;
    for ii in result:
        chk=0
        for jj in ii:
            chk=chk+1
            if jj>val:
                val =jj
                probableClass = chk
    totalExperimented = totalExperimented +1
    if val >= 0.90:

        print('Photo Name       : ', className)
        print('predicted person : ', classes[probableClass - 1])
        print('Probability value: ', val)
        #print('Match with       : ', probableClass)
        truePositive = truePositive + 1
    else:
        print('Photo Name       : ', className)
        print('predicted person : Unknown')
        print('Probability value: ', val)
        #print('Match with       : ', probableClass)

    print()
    print()
    # for i in m1:
    #     if i == m:
    #         break
    #     j = j + 1




    # print(m)


    # ans= m.argmax(axis=0)
    # print(m)

    # print(i)

# print(decisionMartrix)
# predans = 0
# totaltest = 0
# row = 0
# ii=0
# jj=0
# for i in decisionMartrix:
#     jj=0
#     for j in i:
#         if jj==ii:
#             predans = predans + decisionMartrix[ii][jj]
#             totaltest = totaltest + decisionMartrix[ii][jj]
#         else:
#             totaltest = totaltest + decisionMartrix[ii][jj]
#         jj = jj+1
#     ii= ii+1

# ans = float(truePositive * 100) / totalExperimented
# print ("Accuracy = ", ans)
print('Total Predicted    : ', truePositive)
print('Unknown predicted  : ', totalExperimented-truePositive)
