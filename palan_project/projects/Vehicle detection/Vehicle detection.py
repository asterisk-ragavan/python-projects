import cv2
vehiclexml=cv2.CascadeClassifier('vehicle.xml')

def detection(frame):
    vehicle=vehiclexml.detectMultiScale(frame,1.15,4)
    for (x,y,w,h) in vehicle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
        cv2.putText(frame,'vehicle detected.!',(x+w,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),thickness=2)

    return frame

def capturescreen():
    realtimevideo=cv2.VideoCapture(0)
    while realtimevideo.isOpened():
        ret,frame=realtimevideo.read()
        controlkey=cv2.waitKey(1)
        if ret:
            vehicleframe=detection(frame)
            cv2.imshow('vehicle detection',vehicleframe)
        else:
            break
        if controlkey==ord('q'):
            break


    realtimevideo.release()
    cv2.destroyAllWindows()

capturescreen()











kpi = [(1,10,19),(2,11,20),(3,12,21),(4,13,22),(5,14,23),(6,15,24),(7,16,25),(8,17,26),(9,18)]
alp = 'abcdefghijklmnopqrstuvwxyz'
text = "sak"
sum =0
for i in text:
    for l1 in kpi:
        l2c=1
        for l2 in l1:
            if i == alp[l2-1]:
                sum = sum + l2c
            l2c=l2c+1
        l2c=1
print(sum)