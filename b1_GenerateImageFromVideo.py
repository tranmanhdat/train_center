import cv2
import numpy as np
import sys
import argparse
from __config__ import *
from skimage.measure import compare_ssim

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-p', '--prefix')
args = parser.parse_args()


prefix = ''
postfix = ''
if part == 'nuadau':
    prefix = 'd{}__'.format(args.prefix)
if part == 'thang_lanephai':
    prefix = 's{}__'.format(args.prefix) # straight
if part == 'trai_lanephai':
    prefix = 'l{}__'.format(args.prefix) # left
if part == 'thang_lanetrai':
    prefix = 's{}__'.format(args.prefix) # straight
    postfix = '__lanetrai'
if part == 'phai_lanephai':
    prefix = 'r{}__'.format(args.prefix) # right
if part == 'trai_lanetrai':
    prefix = 'll{}__'.format(args.prefix) # re trai, lane trai

cap = cv2.VideoCapture(args.input)

if not cap.isOpened():
    print('Unable to read camera feed')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print('frame_width : ' + str(frame_width))
print('frame_height : ' + str(frame_height))

skip = True
add = 0
k = 0
t = 15
count = 0
last_frame = None
while True:
    ret, frame = cap.read()

    save = True

    if ret:

        # Display the resulting frame
        cv2.imshow('frame', frame)

        key = cv2.waitKey(t)
        if key == 27:
            break
        elif key == ord('s'):
            print('start saving')
            skip = False
        elif key == ord('q'):
            skip = True
            print('stop saving')
        elif key == 82:  # up arrow
            t = t + 5
            print(t)
        elif key == 84:
            t = t - 5
            print(t)
        if not skip:
            k = k + 1
            # if k % 1 == 0:
            if last_frame is not None:
                g1 = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                (score, diff) = compare_ssim(g1, g2, full=True)
                print('ssim', score)
                if score < 1:
                    last_frame = frame
                else:
                    save = False
            else:
                last_frame = frame
            
            if save is True:
                cv2.imwrite(san+'/images/'+part+'/' + prefix + str(k+add) + postfix + '.jpg', frame)
                count = count + 1
                print(count)
    else:
        break

    # When everything done, release the video capture and video write objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
