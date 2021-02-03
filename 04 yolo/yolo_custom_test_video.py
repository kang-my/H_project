import numpy as np
import cv2
import time

min_confidence = 0.5
width = 1280
height = 0
show_ratio = 1.0
title_name = 'Custom Yolo'
# Load Yolo
net = cv2.dnn.readNet("yolo_custom_test_video/model/custom-train-yolo_final.weights", "yolo_custom_test_video/custom/custom-train-yolo.cfg")
classes = []
with open("yolo_custom_test_video/custom/classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
color_lists = np.random.uniform(0, 255, size=(len(classes), 3))

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detectAndDisplay(image):
    h, w = image.shape[:2]
    height = int(h * width / w)
    img = cv2.resize(image, (width, height))

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    


    confidences = []
    names = []
    boxes = []
    colors = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                names.append(classes[class_id])
                colors.append(color_lists[class_id])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = '{} {:,.2%}'.format(names[i], confidences[i])
            color = colors[i]
            print(i, label, x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
            #cv2.putText(img, names[i], (x,y+h), font, 1, color, 2)
         

            # if str(names[i]) == 'lion':
            #     cv2.putText(img,'good',(x,y+h+20),font,1,color,2)
            # elif str(names[i]) == 'dragon':
            #     cv2.putText(img,'good',(x,y+h+20),font,1,color,2)
            # elif str(names[i]) == 'cow':
            #     cv2.putText(img,'good',(x,y+h+20),font,1,color,2)    
            # elif str(names[i]) == 'crane':
            #     cv2.putText(img,'normal',(x,y+h+20),font,1,color,2)
            # elif str(names[i]) == 'sparrow':
            #     cv2.putText(img,'normal',(x,y+h+20),font,1,color,2)
            # elif str(names[i]) == 'Umm':
            #     cv2.putText(img,'bad',(x,y+h+20),font,1,color,2)  

            if names[i]=='dragon' or names[i]=='lion' or names[i]=='cow' :
                cv2.putText(img, 'Good', (x, y+h+12), font, 1, color, 2)
            if names[i]=='crane' or names[i]=='sparrow':
                cv2.putText(img, 'Normal', (x, y+h+12), font, 1, color, 2)
            if names[i]=='Umm':
                cv2.putText(img, 'Bad', (x, y+h+12), font, 1, color, 2)  

            if cv2.waitKey(1) == ord('c'):
                cv2.IMREAD_UNCHANGED
                cv2.imwrite("image/cap01"+".png",img)
                cv2.waitKey(0)
                
        
    

    cv2.imshow(title_name, img)

capture = cv2.VideoCapture(0)
time.sleep(2.0)
if not capture.isOpened:
    print('### Error opening video ###')
    exit(0)


while True:
    ret, frame = capture.read()
    if frame is None:
        print('### No more frame ###')
        capture.release()
        break
    detectAndDisplay(frame)
    # 'q'를 누르면 카메라 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()
