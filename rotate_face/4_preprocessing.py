import os
import numpy as np

def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

def normalize_blinks(num_blinks, time_open, u_time_open, sigma_time_open, time_close, u_time_close, sigma_time_close, max_ear, u_max_ear, sigma_max_ear, min_ear, u_min_ear,sigma_min_ear):
    # input is the blinking features as well as their mean and std, output is a [num_blinksx4] matrix as the normalized blinks
    normalized_blinks = np.zeros([num_blinks, 4])
    normalized_time_open = (time_open[0:num_blinks] - u_time_open) / sigma_time_open
    normalized_blinks[:, 0] = normalized_time_open

    normalized_time_close = (time_close[0:num_blinks]  - u_time_close) / sigma_time_close
    normalized_blinks[:, 1] = normalized_time_close

    normalized_max_ear = (max_ear[0:num_blinks]  - u_max_ear) / sigma_max_ear
    normalized_blinks[:, 2] = normalized_max_ear

    normalized_min_ear = (min_ear[0:num_blinks]  - u_min_ear) / sigma_min_ear

    return normalized_blinks


def unroll_in_time(in_data, window_size, stride):
    # in_data is [n,4]            out_data is [N,Window_size,4]
    n = len(in_data)
    if n <= window_size:
        out_data = np.zeros([1, window_size, 4])
        out_data[0, -n:, :] = in_data
        return out_data
    else:
        N = ((n - window_size) // stride) + 1
        out_data = np.zeros([N, window_size, 4])
        for i in range(N):
            if i * stride + window_size <= n:
                out_data[i, :, :] = in_data[i * stride:i * stride + window_size, :]
            else:  # this line should not ever be executed because of the for mula used above N is the exact time the loop is executed
                break

        return out_data

def gen(folder_list,window_size,stride,path1,len_file):
    lenx = 0
    for ID, folder in enumerate(folder_list):
        print("#########\n")
        print(str(ID) + '-' + str(folder) + '\n')
        print("#########\n")
        files_per_person = os.listdir(path1 + '/' + folder)
        for txt_file in files_per_person:
            if txt_file == 'alert.txt':
                blinksTXT = path1 + '/' + folder + '/' + txt_file
                time_open = np.loadtxt(blinksTXT, usecols=1)
                time_close = np.loadtxt(blinksTXT, usecols=2)
                max_ear = np.loadtxt(blinksTXT, usecols=3)
                min_ear = np.loadtxt(blinksTXT, usecols=4)
                num_blinks = len(time_open)
                bunch_size = num_blinks // 3  # one third used for baselining
                remained_size = num_blinks - bunch_size
                # Using the last bunch_size number of blinks to calculate mean and std
                u_time_open = np.mean(time_open[-bunch_size:])
                sigma_time_open = np.std(time_open[-bunch_size:])
                if sigma_time_open == 0:
                    sigma_time_open = np.std(time_open)

                u_time_close = np.mean(time_close[-bunch_size:])
                sigma_time_close = np.std(time_close[-bunch_size:])
                if sigma_time_close == 0:
                    sigma_time_close = np.std(time_close)

                u_max_ear = np.mean(max_ear[-bunch_size:])
                sigma_max_ear = np.std(max_ear[-bunch_size:])
                if sigma_max_ear == 0:
                    sigma_max_ear = np.std(max_ear)

                u_min_ear = np.mean(min_ear[-bunch_size:])
                sigma_min_ear = np.std(min_ear[-bunch_size:])
                if sigma_min_ear == 0:
                    sigma_min_ear = np.std(min_ear)

                normalized_blinks = normalize_blinks(num_blinks, time_open, u_time_open, sigma_time_open, time_close, u_time_close, sigma_time_close, max_ear, u_max_ear, sigma_max_ear, min_ear, u_min_ear,sigma_min_ear)

                alert_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                # sweep a window over the blinks to chunk
                alert_labels = 0 * np.ones([len(alert_blink_unrolled), 1])
                lenx += len(alert_blink_unrolled)
                with open(len_file, 'ab') as f_handle:
                    f_handle.write(b'\n')
                    np.savetxt(f_handle,[lenx], delimiter=', ', newline=' ',fmt='%d')


            if txt_file == 'semisleepy.txt':
                blinksTXT = path1 + '/' + folder + '/' + txt_file
                time_open = np.loadtxt(blinksTXT, usecols=1)
                time_close = np.loadtxt(blinksTXT, usecols=2)
                max_ear = np.loadtxt(blinksTXT, usecols=3)
                min_ear = np.loadtxt(blinksTXT, usecols=4)
                num_blinks = len(time_open)

                normalized_blinks = normalize_blinks(num_blinks, time_open, u_time_open, sigma_time_open, time_close, u_time_close, sigma_time_close, max_ear, u_max_ear, sigma_max_ear, min_ear, u_min_ear,sigma_min_ear)

                semi_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                semi_labels = 1 * np.ones([len(semi_blink_unrolled), 1])
                lenx += len(semi_blink_unrolled)
                with open(len_file, 'ab') as f_handle:
                    f_handle.write(b'\n')
                    np.savetxt(f_handle,[lenx], delimiter=', ', newline=' ',fmt='%d')


            if txt_file == 'sleepy.txt':
                blinksTXT = path1 + '/' + folder + '/' + txt_file
                time_open = np.loadtxt(blinksTXT, usecols=1)
                time_close = np.loadtxt(blinksTXT, usecols=2)
                max_ear = np.loadtxt(blinksTXT, usecols=3)
                min_ear = np.loadtxt(blinksTXT, usecols=4)
                num_blinks = len(time_open)

                normalized_blinks = normalize_blinks(num_blinks, time_open, u_time_open, sigma_time_open, time_close, u_time_close, sigma_time_close, max_ear, u_max_ear, sigma_max_ear, min_ear, u_min_ear,sigma_min_ear)

                sleepy_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                sleepy_labels = 2 * np.ones([len(sleepy_blink_unrolled), 1])
                lenx += len(sleepy_blink_unrolled)
                with open(len_file, 'ab') as f_handle:
                    f_handle.write(b'\n')
                    np.savetxt(f_handle,[lenx], delimiter=', ', newline=' ',fmt='%d')

        tempX = np.concatenate((alert_blink_unrolled, semi_blink_unrolled, sleepy_blink_unrolled), axis=0)
        tempY = np.concatenate((alert_labels, semi_labels, sleepy_labels), axis=0)
        if ID > 0:
            output = np.concatenate((output, tempX), axis=0)
            labels = np.concatenate((labels, tempY), axis=0)
        else:
            output = tempX
            labels = tempY
    return output,labels



def Preprocess(path1,window_size,stride,test_fold,len_file):
    #path1 is the address to the folder of all subjects, each subject has three txt files for alert, semisleepy and sleepy levels
    #window_size decides the length of blink sequence
    #stride is the step by which the moving windo slides over consecutive blinks to generate the sequences
    #test_fold is the number of fold that is picked as test and uses the other folds as training
    #output=[N,T,F]
    path=path1
    folds_list = os.listdir(path1)
    for f, fold in enumerate(folds_list):
        print(fold)
        path1 = path + '/' + fold
        folder_list = os.listdir(path1)
        if fold==test_fold:
            outTest,labelTest=gen(folder_list,window_size,stride,path1,len_file)
            print("Not this fold ;)")
            continue
        for ID,folder in enumerate(folder_list):
            print("#########\n")
            print(str(ID)+'-'+ str(folder)+'\n')
            print("#########\n")
            files_per_person = os.listdir(path1 + '/' + folder)
            for txt_file in files_per_person:
                if txt_file=='alert.txt':
                    blinksTXT = path1 + '/' + folder + '/' + txt_file
                    time_open = np.loadtxt(blinksTXT, usecols=1)
                    time_close = np.loadtxt(blinksTXT, usecols=2)
                    max_ear = np.loadtxt(blinksTXT, usecols=3)
                    min_ear = np.loadtxt(blinksTXT, usecols=4)
                    num_blinks=len(time_open)
                    bunch_size=num_blinks // 3   #one third used for baselining
                    remained_size=num_blinks-bunch_size
                    # Using the last bunch_size number of blinks to calculate mean and std
                    u_time_open=np.mean(time_open[-bunch_size:])
                    sigma_time_open=np.std(time_open[-bunch_size:])
                    if sigma_time_open==0:
                        sigma_time_open=np.std(time_open)

                    u_time_close=np.mean(time_close[-bunch_size:])
                    sigma_time_close=np.std(time_close[-bunch_size:])
                    if sigma_time_close==0:
                        sigma_time_close=np.std(time_close)

                    u_max_ear = np.mean(max_ear[-bunch_size:])
                    sigma_max_ear = np.std(max_ear[-bunch_size:])
                    if sigma_max_ear == 0:
                        sigma_max_ear = np.std(max_ear)

                    u_min_ear = np.mean(min_ear[-bunch_size:])
                    sigma_min_ear = np.std(min_ear[-bunch_size:])
                    if sigma_min_ear == 0:
                        sigma_min_ear = np.std(min_ear)

                    normalized_blinks = normalize_blinks(num_blinks, time_open, u_time_open, sigma_time_open, time_close, u_time_close, sigma_time_close, max_ear, u_max_ear, sigma_max_ear, min_ear, u_min_ear,sigma_min_ear)


                    alert_blink_unrolled=unroll_in_time(normalized_blinks,window_size,stride)
                    # sweep a window over the blinks to chunk
                    alert_labels = 0 * np.ones([len(alert_blink_unrolled), 1])

                if txt_file=='semisleepy.txt':
                    blinksTXT = path1 + '/' + folder + '/' + txt_file
                    time_open = np.loadtxt(blinksTXT, usecols=1)
                    time_close = np.loadtxt(blinksTXT, usecols=2)
                    max_ear = np.loadtxt(blinksTXT, usecols=3)
                    min_ear = np.loadtxt(blinksTXT, usecols=4)
                    num_blinks = len(time_open)

                    normalized_blinks = normalize_blinks(num_blinks, time_open, u_time_open, sigma_time_open, time_close, u_time_close, sigma_time_close, max_ear, u_max_ear, sigma_max_ear, min_ear, u_min_ear,sigma_min_ear)

                    semi_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                    semi_labels = 1* np.ones([len(semi_blink_unrolled), 1])

                if txt_file == 'sleepy.txt':
                    blinksTXT = path1 + '/' + folder + '/' + txt_file
                    time_open = np.loadtxt(blinksTXT, usecols=1)
                    time_close = np.loadtxt(blinksTXT, usecols=2)
                    max_ear = np.loadtxt(blinksTXT, usecols=3)
                    min_ear = np.loadtxt(blinksTXT, usecols=4)
                    num_blinks = len(time_open)

                    normalized_blinks = normalize_blinks(num_blinks, time_open, u_time_open, sigma_time_open, time_close, u_time_close, sigma_time_close, max_ear, u_max_ear, sigma_max_ear, min_ear, u_min_ear,sigma_min_ear)

                    sleepy_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                    sleepy_labels=2*np.ones([len(sleepy_blink_unrolled),1])


            tempX=np.concatenate((alert_blink_unrolled,semi_blink_unrolled,sleepy_blink_unrolled),axis=0)
            tempY = np.concatenate((alert_labels, semi_labels, sleepy_labels), axis=0)
            if test_fold!="Fold1":
                start=0
            else:
                start=1
            if f !=start  or ID>0:
                output=np.concatenate((output,tempX),axis=0)
                labels=np.concatenate((labels,tempY),axis=0)
            else:
                output=tempX
                labels=tempY

    output,labels=unison_shuffled_copies(output,labels)
    print('We have %d training datapoints!!!' %len(labels))
    print('We have %d test datapoints!!!' % len(labelTest))
    print('We have in TOTAL %d datapoints!!!' % (len(labelTest)+len(labels)))
    return output,labels,outTest,labelTest

##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
path1='drowsiness_data'
stride = 2     
for window_size in [3,5,10,20,30,60,90]:              #[10,20,30,60,90]
    for i in [1,2,3,4,5]:
        test_fold = 'Fold'+str(i)                                
        Training   = './data_preprocess/4_Blinks_'+str(window_size)+'_'+test_fold+'.npy'
        LabelsTrain= './data_preprocess/4_Labels_'+str(window_size)+'_'+test_fold+'.npy'
        Testing    = './data_preprocess/4_BlinksTest_'+str(window_size)+'_'+test_fold+'.npy'
        LabelsTest = './data_preprocess/4_LabelsTest_'+str(window_size)+'_'+test_fold+'.npy'
        len_file = './data_preprocess/4_'+str(window_size)+'_'+test_fold+'.txt'

        ######################################################################
        # Preprocess(path,test_fold,duration,stride)
        blinks,labels,blinksTest,labelTest=Preprocess(path1,window_size,stride,test_fold,len_file)
        np.save(open(Training,'wb'),blinks)
        np.save(open(LabelsTrain, 'wb'),labels)
        np.save(open(Testing, 'wb'),blinksTest)
        np.save(open(LabelsTest, 'wb'),labelTest)
# # --------------------------------------------------------------

# for window_size in [3,5,10,20]:              #[10,20,30,60,90]
#     folder_list = os.listdir("mytest")
#     Testing = './data_preprocess/BlinksTest_'+str(window_size)+'.npy'
#     LabelsTest = './data_preprocess/LabelsTest_'+str(window_size)+'.npy'
#     len_file = './data_preprocess/'+str(window_size)+'.txt'
#     outTest,labelTest=gen(folder_list,window_size,stride,'mytest',len_file)
#     np.save(open(Testing, 'wb'),outTest)
#     np.save(open(LabelsTest, 'wb'),labelTest)         