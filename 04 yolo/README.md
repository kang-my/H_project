## Yolo v3을 이용한 눈 관상 판별
![yolodemoimg](https://user-images.githubusercontent.com/78400692/106859852-f8689400-6706-11eb-9190-dd46a05e2630.png)


### 세부 내용
 - 사용언어: Python [3.6.9]
 - 개발환경: Google Colaboratory(GUI) (OpenCV 4.1.2, Yolo-v3 darknet)
            Visual code studio(1.52.1) (numpy 1.19.4)
 - 사용부품: Logitec HD Webcam

### 기능  
- 사람의 얼굴에서 눈을 인식하여 해당하는 관상을 나타냄

### Data set
- 총 22개의 사진 (이미지 한 장당 3~4개의 데이터 포함)
- 2개의 test set
- 20개의 train set
- Learning_rate = 0,001
- max_batches = 1200 ( class : 6 * 2000)
