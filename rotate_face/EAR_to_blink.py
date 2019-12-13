import os
import numpy as np
from scipy.ndimage.interpolation import shift
import pickle

def REVERSE(status, IF_Closed_Eyes):
    reverse = False
    if (((status == True) and (IF_Closed_Eyes == 1)) or ((status == False) and (IF_Closed_Eyes != 1))):
        reverse = True

    return reverse

def RECORD_Blink(blinks_output_file, time_open, time_close, max_ear, min_ear, left_max, left_min, right_max, right_min, AVGmouth, AVGfaceAngleXs, AVGfaceAngleYs, AVGfaceAngleZs):
    with open(blinks_output_file, 'ab') as f_handle:
        f_handle.write(b'\n')
        np.savetxt(f_handle,[time_open, time_close, max_ear, min_ear, left_max, left_min, right_max, right_min, AVGmouth, AVGfaceAngleXs, AVGfaceAngleYs, AVGfaceAngleZs], delimiter=', ', newline=' ',fmt='%.4f')

def EAR_to_blink(blinks_output_file, leftEARs, rightEARs, mouthEARs, faceAngleXs, faceAngleYs, faceAngleZs):
    EAR_series=np.zeros([13])   
    status = True        #True = eyeOpen    False = eyeClose
    frames_blink = 0
    max_ear = 0.21
    min_ear = 0.21
    left_max = 0.21
    left_min = 0.21
    right_max = 0.21
    right_min = 0.21
    time_close = 0
    time_open = 0

    for i, leftEAR in enumerate(leftEARs):
        rightEAR = rightEARs[i]
        ear = (leftEAR + rightEAR) / 2.0
        EAR_series = shift(EAR_series, -1, cval=ear)
        IF_Closed_Eyes = loaded_svm.predict(EAR_series.reshape(1,-1))
        if (REVERSE(status,IF_Closed_Eyes)):
            if (status == False):
                if (time_open != 0):
                    if (frames_blink > 2):
                        time_close = (frames_blink-2) / FPS
                        AVGmouth = np.mean(mouthEARs[i-frames_blink-2:i-2],axis=0)
                        AVGfaceAngleXs = np.mean(faceAngleXs[i-frames_blink-2:i-2],axis=0)
                        AVGfaceAngleYs = np.mean(faceAngleYs[i-frames_blink-2:i-2],axis=0)
                        AVGfaceAngleZs = np.mean(faceAngleZs[i-frames_blink-2:i-2],axis=0)
                        # print(frames_blink, i)
                        RECORD_Blink(blinks_output_file, time_open, time_close, max_ear, min_ear, left_max, left_min, right_max, right_min, AVGmouth, AVGfaceAngleXs, AVGfaceAngleYs, AVGfaceAngleZs)  
                        max_ear = ear                                
                        left_max = leftEAR
                        right_max = rightEAR
                        status = not status
                        frames_blink = 1
                    else:
                        frames_blink += 1
                else:
                    frames_blink += 1
            else:
                if (frames_blink > 2):
                    time_open = (frames_blink-2) / FPS  
                    min_ear = ear                          
                    left_min = leftEAR
                    right_min = rightEAR            
                    status = not status
                    frames_blink = 1
                else:
                    frames_blink += 1
        else:
            if ((status == True) and (max_ear < ear)):
                max_ear = ear

            if ((status == False) and (ear < min_ear)):
                min_ear = ear

            frames_blink += 1


loaded_svm = pickle.load(open('../Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav', 'rb'))
FPS = 30
# ===============================================================================
# path1='drowsiness_data'
# path=path1
# folds_list = os.listdir(path1)
# for f, fold in enumerate(folds_list):
#     print(fold)
#     path1 = path + '/' + fold
#     folder_list = os.listdir(path1)
#     for ID,folder in enumerate(folder_list):
#         print("#########\n")
#         print(str(ID)+'-'+ str(folder)+'\n')
#         print("#########\n")
#         files_per_person = os.listdir(path1 + '/' + folder)
#         for txt_file in files_per_person:
#             if ((txt_file=='ear_alert.txt') or (txt_file=='ear_semisleepy.txt') or (txt_file=='ear_sleepy.txt')):
#                 blinksTXT = path1 + '/' + folder + '/' + txt_file
#                 blinks_output_file = path1 + '/' + folder + '/' + txt_file.split('_')[1]
#                 leftEARs = np.loadtxt(blinksTXT, usecols=0)
#                 rightEARs = np.loadtxt(blinksTXT, usecols=1)
#                 mouthEARs = np.loadtxt(blinksTXT, usecols=2)
#                 faceAngleXs = np.loadtxt(blinksTXT, usecols=3)
#                 faceAngleYs = np.loadtxt(blinksTXT, usecols=4)
#                 faceAngleZs = np.loadtxt(blinksTXT, usecols=5)
#                 EAR_to_blink(blinks_output_file, leftEARs, rightEARs, mouthEARs, faceAngleXs, faceAngleYs, faceAngleZs)

# =========================================================================================
# files_per_person = os.listdir('./mytest/drowsiness_data')
# for txt_file in files_per_person:
#     if ((txt_file=='ear_alert.txt') or (txt_file=='ear_semisleepy.txt') or (txt_file=='ear_sleepy.txt')):
#         blinksTXT = './mytest/drowsiness_data/' + txt_file
#         blinks_output_file = './mytest/drowsiness_data/' + txt_file.split('_')[1]
#         leftEARs = np.loadtxt(blinksTXT, usecols=0)
#         rightEARs = np.loadtxt(blinksTXT, usecols=1)
#         mouthEARs = np.loadtxt(blinksTXT, usecols=2)
#         faceAngleXs = np.loadtxt(blinksTXT, usecols=3)
#         faceAngleYs = np.loadtxt(blinksTXT, usecols=4)
#         faceAngleZs = np.loadtxt(blinksTXT, usecols=5)
#         EAR_to_blink(blinks_output_file, leftEARs, rightEARs, mouthEARs, faceAngleXs, faceAngleYs, faceAngleZs)

files_per_person = os.listdir('./mytest/drowsiness_data')
for txt_file in files_per_person:
    if (txt_file=='ear_o1-3.txt'):
        blinksTXT = './mytest/drowsiness_data/' + txt_file
        blinks_output_file = './mytest/drowsiness_data/' + txt_file.split('_')[1]
        leftEARs = np.loadtxt(blinksTXT, usecols=0)
        rightEARs = np.loadtxt(blinksTXT, usecols=1)
        mouthEARs = np.loadtxt(blinksTXT, usecols=2)
        faceAngleXs = np.loadtxt(blinksTXT, usecols=3)
        faceAngleYs = np.loadtxt(blinksTXT, usecols=4)
        faceAngleZs = np.loadtxt(blinksTXT, usecols=5)
        EAR_to_blink(blinks_output_file, leftEARs, rightEARs, mouthEARs, faceAngleXs, faceAngleYs, faceAngleZs)