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

def create_model(window_size):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(window_size, 10)),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(512, activation=tf.nn.relu),      
        keras.layers.Dense(3, activation=tf.nn.softmax)
        ])
    #Model I
    model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.005),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    #Model II
    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #         loss=tf.keras.losses.sparse_categorical_crossentropy,
    #         metrics=['accuracy'])

    return model

for window_size in [5]:              
    for i in [0,1,2,3,4]: 
        print ('*************************\n',i,'\n************************\nCreate model')
        Blinks = np.load('./data_preprocess/Blinks_'+str(window_size)+'_Fold'+str(i+1)+'.npy')
        Labels = np.load('./data_preprocess/Labels_'+str(window_size)+'_Fold'+str(i+1)+'.npy')
        BlinksTest = np.load('./data_preprocess/BlinksTest_'+str(window_size)+'_Fold'+str(i+1)+'.npy')
        LabelsTest = np.load('./data_preprocess/LabelsTest_'+str(window_size)+'_Fold'+str(i+1)+'.npy')

        lenx = np.loadtxt('./data_preprocess/'+str(window_size)+'_Fold'+str(i+1)+'.txt', usecols=0)
        # print (len (Labels),len(LabelsTest),lenx)
        model = create_model(window_size)

        # # Restore_models
        # model = keras.models.load_model('./my_model/III_'+str(window_size)+'_model'+str(i+1)+'.h5')
        # model.summary()
        # model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.01),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        # BlinksTest = np.load('./data_preprocess/BlinksTest_'+str(window_size)+'.npy')
        # LabelsTest = np.load('./data_preprocess/LabelsTest_'+str(window_size)+'.npy')
        # lenx = np.loadtxt('./data_preprocess/'+str(window_size)+'.txt', usecols=0)

        print('-----------------Training-------------------')
        # model.fit(Blinks, np.squeeze(np.asarray(Labels)), epochs=80)
        model.fit(Blinks, Labels, batch_size=64, epochs=80)

        print('-------------------Test--------------------')
        filetxt = ('./my_result/result_ear_'+str(window_size)+'_model'+str(i+1)+'.txt')
        filetxt1 = ('./my_result/result_video_'+str(window_size)+'_model'+str(i+1)+'.txt')
        predictions = model.predict(BlinksTest)
        prediction = np.ones([len(predictions), 1])
        for ii in range(len(predictions)):
            prediction[ii] = np.argmax(predictions[ii])

        predict = np.squeeze(np.asarray(prediction))
        accV = 0
        start_blink = 0
        for iii, end_blink in enumerate(lenx):
            # pre = round(np.mean(predict[int(start_blink):int(end_blink)]))
            # lab = round(np.mean(LabelsTest[int(start_blink):int(end_blink)]))
            pre = vote_acc_for_video(np.mean(predict[int(start_blink):int(end_blink)]))
            lab = int(np.mean(LabelsTest[int(start_blink):int(end_blink)]))
            if (pre == lab):
                accV += 1
            with open(filetxt1, "a") as text_file:
                text_file.writelines(str(iii+1)+' Predict: '+str(pre)+ '_'+str(np.mean(predict[int(start_blink):int(end_blink)]))+' Lable true: '+str(lab)+'\n')

            start_blink = end_blink
        with open(filetxt1, "a") as text_file:
            text_file.writelines(str(i+1)+': Test accuracy for video:' +str(accV/len(lenx))+'\n')

        for ii in range(len(predictions)):
            with open(filetxt, "a") as text_file:
                text_file.writelines(str(ii)+' Predict: '+str(np.argmax(predictions[ii]))+ ' Lable true: '+str(LabelsTest[ii])+'\n')
            print (ii,' Predict: ',np.argmax(predictions[ii]), ' Lable true: ',LabelsTest[ii])
        test_loss, test_acc = model.evaluate(BlinksTest, LabelsTest)
        with open(filetxt, "a") as text_file:
            text_file.writelines(str(i+1)+': Test accuracy for blink:' +str(test_acc) +' Loss: ' +str(test_loss)+'\n')
            text_file.writelines('//-----------------------------------------------------------------------------------------\n')

        with open('./my_result/total.csv', "a") as text_file:
            text_file.writelines(str(window_size)+'_'+str(i+1)+', Accuracy for video,' +str(np.round(accV/len(lenx),3))+', Accuracy for ear,' +str(np.round(test_acc,3))+'\n')

        print ('Saving...')
        model.save('./my_model/'+str(window_size)+'_model'+str(i+1)+'.h5')
        print('\n----------------------------------------------------------------------------------------------\n\n')

