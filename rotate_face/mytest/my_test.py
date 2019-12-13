from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np

def vote_acc_for_video(pre):
    if (np.round(pre,5) < 0.66667):
        return 0
    if (np.round(pre,5) <= 1.33333):
        return 1
    return 2

lb = ['Alert','Low vigilant','Drowsy']
for window_size in [5]:              
    for i in [0,1,2,3,4]: 
        model = keras.models.load_model('../my_model/'+str(window_size)+'_model'+str(i+1)+'.h5')
        # model.summary()
        model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.005),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        BlinksTest = np.load('./data_preprocess/BlinksTest_o1-3_'+str(window_size)+'.npy')
        # LabelsTest = np.load('./data_preprocess/LabelsTest_o2_'+str(window_size)+'.npy')
        lenx = np.loadtxt('./data_preprocess/len_o1-3.txt', usecols=0)
        
        filetxt = ('./result/result_ear_o1-3.txt')
        filetxt1 = ('./result/result_video_o1-3.txt')
        predictions = model.predict(BlinksTest)
        prediction = np.ones([len(predictions), 1])
        for ii in range(len(predictions)):
            prediction[ii] = np.argmax(predictions[ii])

        predict = np.squeeze(np.asarray(prediction))
        accV = 0
        start_blink = 0
        for iii, end_blink in enumerate([lenx]):
            # pre = round(np.mean(predict[int(start_blink):int(end_blink)]))
            # lab = round(np.mean(LabelsTest[int(start_blink):int(end_blink)]))
            pre = vote_acc_for_video(np.mean(predict[int(start_blink):int(end_blink)]))
            # lab = int(np.mean(LabelsTest[int(start_blink):int(end_blink)]))
            # if (pre == lab):
            #     accV += 1

            with open(filetxt1, "a") as text_file:
                text_file.writelines('model' + str(i) + ': '+' Predict: '+lb[pre]+'\n')
            print('model' + str(i) + ': '+' Predict: '+lb[pre]+'\n')

            start_blink = end_blink
        # with open(filetxt1, "a") as text_file:
        #     text_file.writelines(str(i+1)+': Test accuracy for video:' +str(accV/len(lenx))+'\n')

        with open(filetxt, "a") as text_file:
            text_file.writelines('model' + str(i) + '-----------------------------\n')
        for ii in range(len(predictions)):
            with open(filetxt, "a") as text_file:
                text_file.writelines(str(ii)+' Predict: '+str(np.argmax(predictions[ii]))+'\n')
        print (predictions)

            # print (ii,' Predict: ',np.argmax(predictions[ii]), ' Lable true: ',LabelsTest[ii])
        # test_loss, test_acc = model.evaluate(BlinksTest, LabelsTest)
        # with open(filetxt, "a") as text_file:
        #     text_file.writelines(str(i+1)+': Test accuracy for blink:' +str(test_acc) +' Loss: ' +str(test_loss)+'\n')
        #     text_file.writelines('//-----------------------------------------------------------------------------------------\n')

        # with open('./result/total.csv', "a") as text_file:
        #     text_file.writelines('o2_'+str(window_size)+'_'+str(i+1)+', Accuracy for video,' +str(np.round(accV/len(lenx),3))+', Accuracy for ear,' +str(np.round(test_acc,3))+'\n')
