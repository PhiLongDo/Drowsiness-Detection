from __future__ import print_function

import cv2
import dlib

# import the necessary packages
from scipy.spatial import distance as dist
import time
from imutils.video import FileVideoStream
import imutils
from imutils import face_utils
# from imutils.face_utils import FaceAligner
# from imutils.face_utils import rect_to_bb
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import*
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage.interpolation import shift
import pickle

import headpose

# -----------------------------------------------------
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

#---------------------------------------------------------------
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    if C<0.1:           #practical finetuning due to possible numerical issue as a result of optical flow
        ear=0.3
    else:
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
    if ear>0.45:        #practical finetuning due to possible numerical issue as a result of optical flow
        ear=0.45

    # return the eye aspect ratio
    return ear

# ---------------------------------------------------------------
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[14], mouth[18])

    C = dist.euclidean(mouth[12], mouth[16])

    if C<0.1:           #practical finetuning
        mar=0.2
    else:
        # compute the mouth aspect ratio
        mar = (A ) / (C)

    # return the mouth aspect ratio
    return mar

# #-----------------------------------------------------------
# def REVERSE(status, IF_Closed_Eyes):
#     reverse = False
#     if (((status == True) and (IF_Closed_Eyes == 1)) or ((status == False) and (IF_Closed_Eyes != 1))):
#         reverse = True

#     return reverse

#-----------------------------------------------------------
# def RECORD_Blink(output_file, TOTAL_BLINKS, time_open, time_close, max_ear, min_ear, left_max, left_min, right_max, right_min):
#     with open(output_file, 'ab') as f_handle:
#         f_handle.write(b'\n')
#         np.savetxt(f_handle,[TOTAL_BLINKS, time_open, time_close, max_ear, min_ear, left_max, left_min, right_max, right_min], delimiter=', ', newline=' ',fmt='%.4f')

# #-----------------------------------------------------------

# def RECORD_EAR(output_file,leftEAR,rightEAR,mouthEAR):
#     with open(output_file, 'ab') as f_handle:
#         f_handle.write(b'\n')
#         np.savetxt(f_handle,[leftEAR, rightEAR, mouthEAR], delimiter=', ', newline=' ',fmt='%.4f')
#-----------------------------------------------------------
def RECORD(output_file, leftEARs, rightEARs, mouthEARs, faceAngleXs, faceAngleYs, faceAngleZs):
    for i in range(len(leftEARs)):
        with open(output_file, 'ab') as f_handle:
            f_handle.write(b'\n')
            np.savetxt(f_handle,[leftEARs[i], rightEARs[i], mouthEARs[i], faceAngleXs[i], faceAngleYs[i], faceAngleZs[i]], delimiter=', ', newline=' ',fmt='%.4f')

#-----------------------------------------------------------
def blink_detector(blinks_output_file,ear_output_file,input_file,angle):
    # -------------------------------------------
    leftEARs  = list()
    rightEARs = list()
    mouthEARs = list()
    faceAngleYs = list()
    faceAngleXs = list()    
    faceAngleZs = list()
    # -------------------------------------------

    EAR_series=np.zeros([13])    
    Frame_series=np.linspace(1,13,13)
    line, = ax.plot(Frame_series,EAR_series)
    lk_params=dict( winSize  = (13,13), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    TOTAL_FRAMES = 0
    # TOTAL_BLINKS = 0    
    FIRST_FACE = True

    # status = True        #True = eyeOpen    False = eyeClose
    # frames_blink = 0
    # max_ear = 0.21
    # min_ear = 0.21
    # left_max = 0.21
    # left_min = 0.21
    # right_max = 0.21
    # right_min = 0.21
    # time_close = 0
    # time_open = 0

    print ("\n------------------------------------------------------\n\t",input_file)

    print("[INFO] loading facial landmark predictor...")

    hpd = headpose.HeadposeDetection()
    face_detect = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')    
    # fa = FaceAligner(predictor, desiredFaceWidth=256)
    loaded_svm = pickle.load(open('../Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav', 'rb'))

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]    
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    video = cv2.VideoCapture(input_file)
    FPS = video.get(cv2.CAP_PROP_FPS)
    video.release()

    print("[INFO] starting video file thread...")
    fvs = FileVideoStream(input_file,queue_size=56).start()

    while fvs.more():
    # while True:
        # Capture frame-by-frame
        frame = fvs.read()
        try:
            if (angle != 0):
                # rotate = 90, -90, 180
                frame = imutils.rotate_bound(frame, angle)                 #                
                
            frame = imutils.resize(frame, width=450)

            TOTAL_FRAMES+=1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = adjust_gamma(gray,gamma=1.5)

            rects = face_detect(gray, 0)

        except:
            # break
            RECORD(ear_output_file, leftEARs, rightEARs, mouthEARs, faceAngleXs, faceAngleYs, faceAngleZs)
            del(leftEARs, rightEARs, mouthEARs, faceAngleXs, faceAngleYs, faceAngleZs)
            break

        if (np.size(rects) != 0):

            shape = predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            # get head pose
            landmarks_2d = hpd.to_numpy(shape).astype(np.double)
            rvec, tvec, cm, dc = hpd.get_headpose(frame, landmarks_2d)
            [rx, ry, rz] = hpd.get_angles(rvec, tvec)

            faceAngleXs.append(rx)
            faceAngleYs.append(ry)
            faceAngleZs.append(rz)
            # print ('\n\t',shape, landmarks_2d, rz)
            # -------------------------------------

            old_gray = gray.copy()
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            MAR = mouth_aspect_ratio(mouth)
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            leftEARs.append(leftEAR) 
            rightEARs.append(rightEAR) 
            mouthEARs.append(MAR)
            # -------------------------------------
            # compute the center of mass for each eye
            # leftEyeCenter = leftEye.mean(axis=0).astype("int")
            # rightEyeCenter = rightEye.mean(axis=0).astype("int")
     
            # # compute the angle between the eye centroids
            # dY = rightEyeCenter[1] - leftEyeCenter[1]
            # dX = rightEyeCenter[0] - leftEyeCenter[0]
            # faceAngleY = np.arctan2(dY, dX)
            # frame = imutils.rotate_bound(frame, -(np.degrees(faceAngleY) - 180))
            # -------------------------------------


            # average the eye aspect ratio together for both eyes
            # ear = (leftEAR + rightEAR) / 2.0
            # EAR_series = shift(EAR_series, -1, cval=ear)
            # IF_Closed_Eyes = loaded_svm.predict(EAR_series.reshape(1,-1))
            # print(IF_Closed_Eyes,'~~~~~')
            if (FIRST_FACE == True):
                FIRST_FACE = False
                # if (int(IF_Closed_Eyes)!=1):
                    # status = True
                    # max_ear = ear                                                    
                    # left_max = leftEAR
                    # right_max = rightEAR
                    # FIRST_FACE = False
                    # frames_blink = 1
            # else:
                # RECORD_EAR(ear_output_file,leftEAR,rightEAR,mouthEAR)
                # ??????????????????????????????????????????????????????????????????
                # if (REVERSE(status,IF_Closed_Eyes)):
                #     if (status == False):
                #         if ((time_open != 0) and (time_close != 0) and (max_ear != min_ear)):
                #             if (frames_blink >= 2):
                #                 TOTAL_BLINKS += 1
                #                 time_close = frames_blink / np.round(FPS)
                #                 # RECORD_Blink(blinks_output_file, TOTAL_BLINKS, time_open, time_close, max_ear, min_ear, left_max, left_min, right_max, right_min)  
                #                 max_ear = ear
                #                 left_max = leftEAR
                #                 right_max = rightEAR
                #                 status = not status
                #                 frames_blink = 1
                #             else:
                #                 frames_blink += 1
                #         frames_blink += 1
                #     else:
                #         time_open = time_close = frames_blink / np.round(FPS)   
                #         min_ear = ear
                #         left_min = leftEAR
                #         right_min = rightEAR            
                #         status = not status
                #         frames_blink = 1
                # else:
                #     if ((status == True) and (max_ear < ear)):
                #         max_ear = ear

                #     if ((status == False) and (ear < min_ear)):
                #         min_ear = ear

                #     frames_blink += 1
                # ?????????????????????????????????????????????????????????????????

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)
            # cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 70),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "mouth: {:.2f}".format(mouthEAR), (10, 90),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # line.set_ydata(EAR_series)
        else:
            st=0
            # st2=0
            if (FIRST_FACE == False):
                # leftEye=leftEye.astype(np.float32)
                # rightEye = rightEye.astype(np.float32)
                # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray,leftEye, None, **lk_params)
                # p2, st2, err2 = cv2.calcOpticalFlowPyrLK(old_gray, gray, rightEye, None, **lk_params)
                shape = shape.astype(np.float32)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray,shape, None, **lk_params)

            # if np.sum(st)+np.sum(st2)==12 and FIRST_FACE==False:
            if (np.sum(st)==68 and FIRST_FACE == False):
                p1 = np.round(p1).astype(np.int)

                landmarks_2d = hpd.to_numpy(p1).astype(np.double)
                rvec, tvec, cm, dc = hpd.get_headpose(frame, landmarks_2d)
                [rx, ry, rz] = hpd.get_angles(rvec, tvec)

                faceAngleXs.append(rx)
                faceAngleYs.append(ry)
                faceAngleZs.append(rz)
                # p2 = np.round(p2).astype(np.int)
                #print(p1)

                # leftEAR = eye_aspect_ratio(p1)
                # rightEAR = eye_aspect_ratio(p2)

                leftEye = p1[lStart:lEnd]
                rightEye = p1[rStart:rEnd]
                mouth = p1[mStart:mEnd]

                MAR = mouth_aspect_ratio(mouth)
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                leftEARs.append(leftEAR) 
                rightEARs.append(rightEAR) 
                mouthEARs.append(MAR)

                # -------------------------------------
                # compute the center of mass for each eye
                # leftEyeCenter = leftEye.mean(axis=0).astype("int")
                # rightEyeCenter = rightEye.mean(axis=0).astype("int")
         
                # compute the angle between the eye centroids
                # dY = rightEyeCenter[1] - leftEyeCenter[1]
                # dX = rightEyeCenter[0] - leftEyeCenter[0]
                # faceAngleY = np.arctan2(dY, dX)
                # -------------------------------------

                # ear = (leftEAR + rightEAR) / 2.0
                # EAR_series = shift(EAR_series, -1, cval=ear)
                #EAR_series[reference_frame] = ear
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)
                old_gray = gray.copy()
                # leftEye = p1
                # rightEye = p2

            # IF_Closed_Eyes = loaded_svm.predict(EAR_series.reshape(1,-1))
            # print(IF_Closed_Eyes,'^^^^^^^^')
            # if (FIRST_FACE == True):
            #     FIRST_FACE = False
                # if (int(IF_Closed_Eyes)!=1):
                #     status = True
                #     max_ear = ear                                
                #     left_max = leftEAR
                #     right_max = rightEAR
                #     FIRST_FACE = False
                #     frames_blink = 1
            # else:
            #     # RECORD_EAR(ear_output_file,leftEAR,rightEAR,mouthEAR)
            #     # ??????????????????????????????????????????????????????????????????
            #     if (REVERSE(status,IF_Closed_Eyes)):
            #         if (status == False):
            #             if ((time_open != 0) and (time_close != 0) and (max_ear != min_ear)):
            #                 if (frames_blink >= 2):
            #                     TOTAL_BLINKS += 1
            #                     time_close = frames_blink / np.round(FPS)
            #                     # RECORD_Blink(blinks_output_file, TOTAL_BLINKS, time_open, time_close, max_ear, min_ear, left_max, left_min, right_max, right_min)  
            #                     max_ear = ear                                
            #                     left_max = leftEAR
            #                     right_max = rightEAR
            #                     status = not status
            #                     frames_blink = 1
            #                 else:
            #                     frames_blink += 1
            #             frames_blink += 1
            #         else:
            #             time_open = time_close = frames_blink / np.round(FPS)   
            #             min_ear = ear                          
            #             left_min = leftEAR
            #             right_min = rightEAR            
            #             status = not status
            #             frames_blink = 1
            #     else:
            #         if ((status == True) and (max_ear < ear)):
            #             max_ear = ear

            #         if ((status == False) and (ear < min_ear)):
            #             min_ear = ear

            #         frames_blink += 1
            #     # ?????????????????????????????????????????????????????????????????

            # # cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 70),
            # #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # line.set_ydata(EAR_series)

        # cv2.putText(frame, "Blinks: {}".format(TOTAL_BLINKS), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "time: {:.2f}".format(TOTAL_FRAMES/FPS/60), (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the video output
        plot_frame.draw()
        cv2.imshow("Frame", frame)

        # Quit video by typing Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fvs.stop()
    # video_capture.release()
    cv2.destroyAllWindows()



top = tk.Tk()
frame1 = Frame(top)
frame1.grid(row=0, column=0)
fig = plt.figure()
ax = fig.add_subplot(111)
plot_frame =FigureCanvasTkAgg(fig, master=frame1)
plot_frame.get_tk_widget().pack(side=tk.BOTTOM, expand=True)
plt.ylim([0.0, 0.5])
plot_frame.draw()

# Demo--------------------------------------------------------------------

# blinks_output_file = './mytest/drowsiness_data/o1-2.txt'  # The text file to write to (for blinks)#
# ear_output_file = './mytest/drowsiness_data/ear_o1-2.txt'  # The text file to write to (for blinks)#
# path = './mytest/dataset/object1_2.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

blinks_output_file = './mytest/drowsiness_data/o1-3.txt'  # The text file to write to (for blinks)#
ear_output_file = './mytest/drowsiness_data/ear_o1-3.txt'  # The text file to write to (for blinks)#
path = './mytest/dataset/object1_3.mp4' # the path to the input video
blink_detector(blinks_output_file,ear_output_file,path,180)
# Demo--------------------------------------------------------------------

# ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/01/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/01/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/01/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold1/01/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/01/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/01/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/01/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/01/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/01/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)
# ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/02/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/02/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/02/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/02/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/02/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/02/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold1/02/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/02/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/02/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/03/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/03/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/03/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold1/03/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/03/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/03/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/03/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/03/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/03/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/04/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/04/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/04/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/04/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/04/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/04/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/04/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/04/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/04/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/05/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/05/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/05/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold1/05/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/05/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/05/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold1/05/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/05/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/05/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/06/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/06/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/06/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/06/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/06/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/06/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/06/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/06/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part1/06/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/07/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/07/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/07/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold1/07/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/07/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/07/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/07/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/07/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/07/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)
# # # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/08/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/08/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/08/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold1/08/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/08/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/08/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold1/08/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/08/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/08/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# # # ===============================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/09/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/09/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/09/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/09/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/09/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/09/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/09/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/09/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/09/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)
# # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/10/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/10/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/10/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold1/10/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/10/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/10/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold1/10/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/10/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/10/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# # ===============================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/11/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/11/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/11/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/11/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/11/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/11/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/11/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/11/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/11/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)
# # ===============================================================================================================
# blinks_output_file = './drowsiness_data/Fold1/12/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/12/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/12/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/12/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/12/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/12/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold1/12/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold1/12/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold1_part2/12/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold2/13/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/13/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/13/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold2/13/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/13/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/13/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold2/13/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/13/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/13/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)
# # # ====================================================================================
# blinks_output_file = './drowsiness_data/Fold2/14/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/14/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/14/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold2/14/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/14/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/14/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/14/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/14/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/14/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)
# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold2/15/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/15/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/15/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/15/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/15/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/15/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/15/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/15/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/15/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)


# ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold2/16/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/16/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/16/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/16/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/16/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/16/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/16/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/16/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/16/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)
# # ======================================================================================
# blinks_output_file = './drowsiness_data/Fold2/17/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/17/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/17/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/17/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/17/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/17/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/17/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/17/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/17/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)
# # ====================================================================================
# blinks_output_file = './drowsiness_data/Fold2/18/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/18/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/18/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/18/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/18/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/18/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/18/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/18/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part1/18/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)



# # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold2/19/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/19/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/19/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold2/19/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/19/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/19/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold2/19/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/19/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/19/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)
# # # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold2/20/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/20/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/20/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/20/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/20/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/20/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/20/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/20/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/20/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)
# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold2/21/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/21/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/21/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold2/21/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/21/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/21/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold2/21/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/21/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/21/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)



# # ============================================================================================
# blinks_output_file = './drowsiness_data/Fold2/22/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/22/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/22/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold2/22/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/22/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/22/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold2/22/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/22/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/22/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)
# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold2/23/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/23/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/23/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/23/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/23/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/23/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/23/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/23/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/23/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)
# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold2/24/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/24/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/24/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/24/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/24/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/24/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold2/24/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold2/24/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold2_part2/24/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# # # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold3/25/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/25/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/25/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold3/25/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/25/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/25/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold3/25/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/25/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/25/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)
# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold3/26/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/26/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/26/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/26/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/26/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/26/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,180)

# blinks_output_file = './drowsiness_data/Fold3/26/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/26/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/26/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold3/27/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/27/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/27/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/27/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/27/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/27/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/27/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/27/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/27/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)


# # # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold3/28/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/28/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/28/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold3/28/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/28/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/28/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold3/28/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/28/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/28/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)
# # =========================================================================================================

# blinks_output_file = './drowsiness_data/Fold3/29/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/29/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/29/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/29/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/29/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/29/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/29/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/29/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/29/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)


# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold3/30/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/30/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/30/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/30/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/30/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/30/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/30/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/30/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part1/30/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold3/31/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/31/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/31/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/31/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/31/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/31/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/31/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/31/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/31/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold3/32/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/32/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/32/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/32/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/32/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/32/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/32/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/32/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/32/10_1.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/32/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/32/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/32/10_2.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold3/33/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/33/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/33/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold3/33/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/33/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/33/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold3/33/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/33/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/33/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold3/34/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/34/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/34/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/34/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/34/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/34/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/34/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/34/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/34/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold3/35/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/35/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/35/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/35/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/35/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/35/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/35/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/35/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/35/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold3/36/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/36/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/36/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/36/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/36/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/36/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold3/36/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold3/36/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold3_part2/36/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)


# # # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# # # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/37/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/37/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/37/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/37/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/37/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/37/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/37/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/37/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/37/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/38/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/38/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/38/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold4/38/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/38/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/38/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold4/38/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/38/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/38/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/39/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/39/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/39/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/39/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/39/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/39/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/39/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/39/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/39/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/40/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/40/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/40/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/40/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/40/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/40/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/40/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/40/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/40/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/41/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/41/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/41/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/41/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/41/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/41/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/41/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/41/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part1/41/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)


# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/42/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/42/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/42/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold4/42/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/42/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/42/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold4/42/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/42/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/42/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)
# # # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/43/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/43/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/43/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold4/43/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/43/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/43/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold4/43/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/43/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/43/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/44/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/44/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/44/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/44/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/44/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/44/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/44/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/44/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/44/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/45/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/45/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/45/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,180)

# blinks_output_file = './drowsiness_data/Fold4/45/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/45/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/45/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,180)

# blinks_output_file = './drowsiness_data/Fold4/45/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/45/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/45/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,180)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/46/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/46/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/46/0.m4v' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/46/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/46/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/46/5.m4v' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/46/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/46/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/46/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/47/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/47/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/47/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold4/47/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/47/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/47/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold4/47/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/47/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/47/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# # =========================================================================================================
# blinks_output_file = './drowsiness_data/Fold4/48/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/48/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/48/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/48/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/48/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/48/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold4/48/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold4/48/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold4_part2/48/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)



# # //////////////////////////////////////////////////////////////////////////////////////////////////////////////

# # ================================================================================================================

# blinks_output_file = './drowsiness_data/Fold5/49/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/49/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/49/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold5/49/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/49/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/49/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold5/49/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/49/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/49/10_1.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold5/49/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/49/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/49/10_2.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold5/50/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/50/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/50/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold5/50/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/50/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/50/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold5/50/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/50/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/50/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)
# # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold5/51/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/51/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/51/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold5/51/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/51/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/51/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold5/51/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/51/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/51/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold5/52/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/52/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/52/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold5/52/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/52/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/52/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold5/52/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/52/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/52/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold5/53/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/53/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/53/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90,)

# blinks_output_file = './drowsiness_data/Fold5/53/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/53/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/53/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90,)

# blinks_output_file = './drowsiness_data/Fold5/53/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/53/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/53/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90,)

# # ==================================================================================================================
# blinks_output_file = './drowsiness_data/Fold5/54/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/54/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/54/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold5/54/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/54/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/54/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold5/54/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/54/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part1/54/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# =================================================================================================================

# blinks_output_file = './drowsiness_data/Fold5/55/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/55/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/55/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold5/55/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/55/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/55/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold5/55/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/55/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/55/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)
# # =================================================================================================================

# blinks_output_file = './drowsiness_data/Fold5/56/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/56/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/56/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold5/56/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/56/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/56/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold5/56/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/56/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/56/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)
# # =================================================================================================================

# blinks_output_file = './drowsiness_data/Fold5/57/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/57/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/57/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold5/57/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/57/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/57/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold5/57/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/57/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/57/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# # =================================================================================================================

# blinks_output_file = './drowsiness_data/Fold5/58/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/58/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/58/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold5/58/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/58/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/58/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './drowsiness_data/Fold5/58/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/58/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/58/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)
# # =================================================================================================================

# blinks_output_file = './drowsiness_data/Fold5/59/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/59/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/59/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold5/59/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/59/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/59/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)

# blinks_output_file = './drowsiness_data/Fold5/59/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/59/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/59/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,90)


# # =================================================================================================================

# blinks_output_file = './drowsiness_data/Fold5/60/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/60/ear_alert.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/60/0.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold5/60/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/60/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/60/5.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)

# blinks_output_file = './drowsiness_data/Fold5/60/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './drowsiness_data/Fold5/60/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = '../Dataset/Fold5_part2/60/10.mov' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,0)






















# =================================================================================================================

# blinks_output_file = './mytest/drowsiness_data/alert.txt'  # The text file to write to (for blinks)#
# ear_output_file = './mytest/drowsiness_data/ear_alert.txt'  # The text file to write to (for blinks)#
# path = './mytest/dataset/0.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './mytest/drowsiness_data/semisleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './mytest/drowsiness_data/ear_semisleepy.txt'  # The text file to write to (for blinks)#
# path = './mytest/dataset/5.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)


# blinks_output_file = './mytest/drowsiness_data/sleepy.txt'  # The text file to write to (for blinks)#
# ear_output_file = './mytest/drowsiness_data/ear_sleepy.txt'  # The text file to write to (for blinks)#
# path = './mytest/dataset/10.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

# blinks_output_file = './mytest/drowsiness_data/o2.txt'  # The text file to write to (for blinks)#
# ear_output_file = './mytest/drowsiness_data/ear_o2.txt'  # The text file to write to (for blinks)#
# path = './mytest/dataset/object2.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)


# blinks_output_file = './mytest/drowsiness_data/o1-2.txt'  # The text file to write to (for blinks)#
# ear_output_file = './mytest/drowsiness_data/ear_o1-2.txt'  # The text file to write to (for blinks)#
# path = './mytest/dataset/object1_2.mp4' # the path to the input video
# blink_detector(blinks_output_file,ear_output_file,path,-90)

