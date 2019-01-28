# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import FileVideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import pandas as pd
import statistics
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=True, help="path to input video file")
args = vars(ap.parse_args())

print(args['video']) 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)
 
# start the FPS timer
# fps = FPS().start()

count = 1
columns = ['frame', 'x', 'y']
columns_1 = ['A', 'B', 'C', 'D','E', 'F', 'G', 'H']
columns_2 = ['SD1', 'SD2', 'SD3', 'SD4', 'SD5', 'SD6', 'SD7', 'SD8']
columns_x = ['1_x', '2_x', '3_x', '4_x', '5_x', '6_x', '7_x', '8_x', '9_x', '10_x', '11_x', '12_x', '13_x', '14_x', '15_x', '16_x', '17_x', '18_x', '19_x', '20_x', '21_x', '22_x', '23_x', '24_x', '25_x', '26_x', '27_x', '28_x', '29_x', '30_x', '31_x', '32_x', '33_x', '34_x', '35_x', '36_x', '37_x', '38_x', '39_x', '40_x', '41_x', '42_x', '43_x', '44_x', '45_x', '46_x', '47_x', '48_x', '49_x', '50_x', '51_x', '52_x', '53_x', '54_x', '55_x', '56_x', '57_x', '58_x', '59_x', '60_x', '61_x', '62_x', '63_x', '64_x', '65_x', '66_x', '67_x', '68_x']
columns_y = ['1_y', '2_y', '3_y', '4_y', '5_y', '6_y', '7_y', '8_y', '9_y', '10_y', '11_y', '12_y', '13_y', '14_y', '15_y', '16_y', '17_y', '18_y', '19_y', '20_y', '21_y', '22_y', '23_y', '24_y', '25_y', '26_y', '27_y', '28_y', '29_y', '30_y', '31_y', '32_y', '33_y', '34_y', '35_y', '36_y', '37_y', '38_y', '39_y', '40_y', '41_y', '42_y', '43_y', '44_y', '45_y', '46_y', '47_y', '48_y', '49_y', '50_y', '51_y', '52_y', '53_y', '54_y', '55_y', '56_y', '57_y', '58_y', '59_y', '60_y', '61_y', '62_y', '63_y', '64_y', '65_y', '66_y', '67_y', '68_y']
df = pd.DataFrame(columns = columns)
distance_df = pd.DataFrame(columns = columns_1)
whole_concated_df = pd.DataFrame(columns = columns_x + columns_y + ['frame'])
# aggregated_df = pd.DataFrame(columns = columns_2) 
# loop over the frames from the video stream
while fvs.more():
    list = []
    # grab the frame from the threaded video stream, resize it to have a maximum width of 400 pixels, and convert it to grayscale
    frame = fvs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    # loop over the face detections, here we only have one face in the frame
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array

        # maybe this is what we want, includes 68(x, y)-coordinates of the facial landmark regions
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        count_list = np.ones((68,1))
        new_shape = np.hstack((count_list,shape))
        for row in new_shape:
            row[0] = count
            # print(new_shape)
            df.append[row]

        new_df = pd.DataFrame(new_shape.reshape(-1, 3), columns = columns)
        print(count)

        # we can do reshape here
        # seperate x and y, 
        x_part_list = new_df.x.tolist()

        x_part_df = pd.DataFrame(columns=columns_x)
        x_part_df = x_part_df.append(pd.Series(x_part_list, index=columns_x), ignore_index=True)

        y_part_list = new_df.y.tolist()

        y_part_df = pd.DataFrame(columns=columns_y)
        y_part_df = y_part_df.append(pd.Series(y_part_list, index=columns_y), ignore_index=True)
        concated_df = pd.concat([x_part_df, y_part_df], axis = 1)
        concated_df['frame'] = count
        whole_concated_df = whole_concated_df.append(concated_df)
        print(concated_df)
        new_x_part_df.to_csv('new_results/' + args['video'] + '.csv')

        print('every new df is:')
        print(new_df)

        df = df.append(new_df)
        print(new_df.iloc[10]['x'])

        add frame count, maybe no need for that
        list.append(count)
        firstly add distance of 11, 12
        list.append(np.sqrt( (new_df.iloc[10]['x'] - new_df.iloc[11]['x'])**2 + (new_df.iloc[10]['y'] - new_df.iloc[11]['y'])**2 ))

        add distance of 9, 10
        list.append(np.sqrt( (new_df.iloc[8]['x'] - new_df.iloc[9]['x'])**2 + (new_df.iloc[8]['y'] - new_df.iloc[9]['y'])**2 ))

        add distance of 2, 18
        list.append(np.sqrt( (new_df.iloc[1]['x'] - new_df.iloc[17]['x'])**2 + (new_df.iloc[1]['y'] - new_df.iloc[17]['y'])**2 ))

        add distance of 1, 17
        list.append(np.sqrt( (new_df.iloc[0]['x'] - new_df.iloc[16]['x'])**2 + (new_df.iloc[0]['y'] - new_df.iloc[16]['y'])**2 ))

        add distance of 11, 21
        list.append(np.sqrt( (new_df.iloc[10]['x'] - new_df.iloc[20]['x'])**2 + (new_df.iloc[10]['y'] - new_df.iloc[20]['y'])**2 ))

        add distance of 9,22
        list.append(np.sqrt( (new_df.iloc[8]['x'] - new_df.iloc[21]['x'])**2 + (new_df.iloc[8]['y'] - new_df.iloc[21]['y'])**2 ))

        add distance of 7, 8
        list.append(np.sqrt( (new_df.iloc[6]['x'] - new_df.iloc[7]['x'])**2 + (new_df.iloc[6]['y'] - new_df.iloc[7]['y'])**2 ))

        add distance of 5, 6
        list.append(np.sqrt( (new_df.iloc[4]['x'] - new_df.iloc[5]['x'])**2 + (new_df.iloc[4]['y'] - new_df.iloc[5]['y'])**2 ))

        # calculate the stand deviation of these values at last
        sd = statistics.stdev(list)
        list.append(sd)
        list_df = pd.DataFrame([list], columns = columns_1)
        # print(list_df)
        distance_df = distance_df.append(list_df)

        save it to pandas, then save it fo a file
        it is working, now we try to add it to dataframe with a special index of count
        np.savetxt('np.csv', shape, fmt='%.2f', delimiter=',')


 
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    
    
    count += 1  
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

# # print(df)
name = 'new_results/' + args['video'] + '.csv'
# print(name)
# print(whole_concated_df.iloc[:,0])
whole_concated_df.to_csv(name)

distance_name = 'distance/' + args['video'] + '.csv'
# set index
distance_df.index = range(1,len(distance_df)+1)

#  discard some data, based on 30 frames 
number = (len(distance_df)+1)//30
discarded_distance_df = distance_df[:(number*30)]
# print(discarded_distance_df)
distance_df.to_csv(distance_name)

mean_df = pd.DataFrame(np.einsum('ijk->ik',discarded_distance_df.values.reshape(-1,30,discarded_distance_df.shape[1]))/30.0, columns=columns_1)
# pd.DataFrame(distance_df.values.reshape(-1,2,distance_df.shape[1]).mean(1))
# distance_df.groupy(np.arange(len(distance_df))//50).mean()
print(mean_df)
std_df = pd.DataFrame(discarded_distance_df.values.reshape(-1,30,discarded_distance_df.shape[1]).std(1), columns=columns_2)
print(std_df)
distance_df.groupy(np.arange(len(distance_df))//50).std()
aggregated_df = pd.concat([mean_df, std_df], axis = 1)
aggregated_df.index = range(1,len(aggregated_df)+1)
print(aggregated_df)
aggregated_name = 'aggregated_30_results/' + args['video'] + '.csv'
aggregated_df.to_csv(aggregated_name)

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()