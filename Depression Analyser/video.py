# Importing all necessary libraries 
import cv2 
from PIL import Image
#import os
import time 
import imagehash

# frame 
currentframe = 0
dataset = cv2.CascadeClassifier('data.xml')

for i in range(1,249):  
    # Read the video from specified path 
    cam = cv2.VideoCapture('videos/a ({}).avi'.format(i)) 
      
    #try: 
    #      
    #    # creating a folder named data 
    #    if not os.path.exists('data'): 
    #        os.makedirs('data') 
    #  
    ## if not created then raise error 
    #except OSError: 
    #    print ('Error: Creating directory of data') 
      
      
    while(True): 
          
        # reading from frame 
        ret,frame = cam.read()
        print(ret) 
          
        if ret: 
            # if video is still left continue creating images
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = dataset.detectMultiScale(gray,1.28)
            for x,y,w,h in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),4)
    
                frame = gray[y:y+h, x:x+w]
    #        face = cv2.resize(face, (64,64))
    
            if currentframe > 2:
                name2 = 'images/image.jpg'
                cv2.imwrite(name2, frame)
                hash = imagehash.average_hash(Image.open('images/image.jpg'))
                otherhash = imagehash.average_hash(Image.open('images/frame'+str(currentframe - 1)+'.jpg'))
                diff = hash - otherhash
#                print("current frame",currentframe-1)
                print("diff",diff)
                if diff > 4:
                    name = 'images/frame' + str(currentframe) + '.jpg'
                    print ('Creating...' + name) 
                  
                    # writing the extracted images 
                    cv2.imwrite(name, frame) 
                  
                    # increasing counter so that it will 
                    # show how many frames are created 
                    currentframe += 1
                    print(currentframe)
                    
            elif currentframe <= 2:
                name = 'images/frame' + str(currentframe) + '.jpg'
                print ('Creating...' + name) 
              
                # writing the extracted images 
                cv2.imwrite(name, frame) 
              
                # increasing counter so that it will 
                # show how many frames are created 
                currentframe += 1
                print(currentframe)
            time.sleep(3)
        else:
            break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 