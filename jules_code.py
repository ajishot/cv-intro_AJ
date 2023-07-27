
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dt_apriltags import Detector
import numpy as np
import matplotlib.animation as animation
import matplotlib.cm as cm
#img = cv2.imread('rov_pool.jpg')

#img = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE)
#plt.imshow(img)
cap = cv2.VideoCapture('AUV_Vid.mkv')

ret, frame = cap.read()

#vcap = cv2.VideoCapture("rtsp://10.29.17.108:8554/test")
#ret, frame = vcap.read()
#plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False
def makegray(img):
    gray = cv2.cvtColor(img,cv2.IMREAD_GRAYSCALE) # convert to grayscale
    return gray

plt.imshow(makegray(frame))
<matplotlib.image.AxesImage at 0x7f1c6e4be0>

def detectedges(img):
    edges = cv2.Canny(img,41, 110, apertureSize=3) # detect edges
    return edges

plt.imshow(detectedges(makegray(frame)))
<matplotlib.image.AxesImage at 0x7f1c6a2ac0>

def makelines(edges):
    lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi/180,
                100,
                minLineLength=100,
                maxLineGap=10,
        ) # detect lines
    return lines
print(makelines(detectedges(makegray(frame))))
[[[1408  807 1745  950]]

 [[1726  887 1893  948]]

 [[1707  933 1918 1022]]

 [[1476  837 1641  907]]

 [[ 856  873  959 1077]]

 [[1573  830 1733  889]]

 [[ 831  822  911  979]]]
def drawline(img):
    grey = makegray(img)
    edges = detectedges(grey)
    lines = makelines(edges)
    
    gradient1 = [None]
    gradient = []
    
    if lines is None:
            pass
    else:
        for line in lines:
            
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
    return(img)
   
        
def get_slopes_intercepts(lines):
    gradient1 = [None]
    gradient = []
    intercepts1 = []
    intercepts = []
    if lines is None:
            pass
    else:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            gradient1[0] = (y2-y1)/(x2-x1)
            intercepts1[0] = (y2/(gradient1[0]*x2))
            if common_member(gradient, gradient1) == False:
                gradient.append(gradient1[0])
                intercepts.append(intercepts1[0])
        

        
    
    return(gradient, intercepts)
def detect_lanes(lines):
    i = 0
    lanes = []
    while i <= len(get_slopes_intercepts(lines)[0]):
        j = 0
        #if parrelel then it is a lane
        while j< i:
            m = round(get_slopes_intercepts(lines)[0][j], 1)
            b = round(get_slopes_intercepts(lines)[1][j], 1)
            a = round(get_slopes_intercepts(lines)[0][i],1)
            c = round(get_slopes_intercepts(lines)[0][i], 1)
            x = (c-b)/(m-a)
            y = m*x+b
            if  m == a:
                lanes.append([lines[j], lines[i]])
                break
            elif x < lines[j].x1 or x > lines[j].x2:
                lanes.append([lines[j], lines[i]])
            j+=1
        #if eventualy going to intersect but not currently intersecting it is a lane
        
        i+=1
count = 0
frequency = 200
newframe = []
while ret:
    if count%frequency == 0:
            newframe.append(drawline(frame))
    count +=1
    ret, frame = cap.read()
    
i = 0
while i < len(newframe):
   
   plt.imshow(newframe[i])
   i+=1
print(len(newframe))
11

animated = [] # for storing the generated images
fig = plt.figure()
for i in range(6):
    animated.append([plt.imshow(newframe[i], cmap=cm.Greys_r,animated=True)])

ani = animation.ArtistAnimation(fig, animated, interval=5000, blit=True,
                                repeat_delay=1000)
plt.show(ani)

#cv2.line(img, (0, 0), (100, 100), (255, 0, 0), 5)
#cv2.rectangle(img, (0, 0), (100, 100), (0, 255, 0), 5)
#cv2.circle(img, (50, 50), 50, (0, 0, 255), 5)
#pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
#pts = pts.reshape((-1, 1, 2))
#cv2.polylines(img, [pts], True, (0, 255, 255), 5)
#cv2.putText(img, 'Hello World!', (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
'''at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
tags = at_detector.detect(img, estimate_tag_pose=False, camera_params=None, tag_size=None)
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for tag in tags:
    for idx in range(len(tag.corners)):
        cv2.line(color_img, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

    cv2.putText(color_img, str(tag.tag_id),
                org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255))
plt.imshow(color_img)'''
"at_detector = Detector(families='tag36h11',\n                       nthreads=1,\n                       quad_decimate=1.0,\n                       quad_sigma=0.0,\n                       refine_edges=1,\n                       decode_sharpening=0.25,\n                       debug=0)\ntags = at_detector.detect(img, estimate_tag_pose=False, camera_params=None, tag_size=None)\ncolor_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)\nfor tag in tags:\n    for idx in range(len(tag.corners)):\n        cv2.line(color_img, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))\n\n    cv2.putText(color_img, str(tag.tag_id),\n                org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),\n                fontFace=cv2.FONT_HERSHEY_SIMPLEX,\n                fontScale=0.8,\n                color=(0, 0, 255))\nplt.imshow(color_img)"