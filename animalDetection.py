# import package
import os 
import base64

try:
    import requests
except:
    os.system('pip install requests')
    import requests

try:
    import cv2
except:
    os.system('pip install opencv-python')
    import cv2

try:
    import numpy as np
except:
    os.system('pip install numpy')
    import numpy as np

try:
    import eel
except:
    os.system('pip install eel')
    import eel

# -- biến toàn cục --
whTarget = 320
confThreshold = 0.5
mnsThreshold = 0.3
color = (0, 0, 255)

detectedImgPath = 'Static/picture/result.jpg'

#tệp chứa tên các lớp mà mô hình có thể phát hiện.
classFiles = '../config/object.names'

#tệp cấu hình cho mô hình YOLO, nơi định nghĩa kiến trúc mạng
modelConfig = '../config/animal.cfg'

# file weights khi không có sẽ tự cài trên cmd tệp trọng số đã được huấn luyện cho mô hình YOLO.
modelWeights = '../config/animal_best.weights'

weights_url = 'https://dl.dropboxusercontent.com/s/9qihxeki243heqw/animal_best.weights?dl=0'


# -- init thư mục chứa html --
eel.init('Static')
print('----------------------------------------------------')
print('-              FINAL SEMESTER PROJECT              -')
print('-          INTRODUCTION TO ANIMAL ANALYSIS         -')
print('-                                                  -')
print('-          2254810297 - Nguyễn Đức Thức            -')
print('-          2254810287 - Nguyễn Trần Chí Trung      -')
print('-          2331540280 - Phan Thị Thanh Thảo        -')
print('-          2254810303 - Nguyễn Ngọc Vinh           -')
print('-          2254810149 - Nguyễn Tiến Minh           -')
print('----------------------------------------------------')


# Mở file để ghi với chế độ nhị phân (wb) vì mình đang phân tích hình ảnh của các con vật có trong hình
def download(url, file_name):
    with open(file_name, "wb") as file:
# nhận yêu cầu
        response = requests.get(url, allow_redirects=True)
# ghi vào tệp
        file.write(response.content)
    

# -- tải file weight -- 
# check xem file weight đã có hay chưa
print('>> Kiểm tra file weights ....', end=' ')
isExist = os.path.isfile(modelWeights)
print(isExist,'!', sep='')

# nếu có rồi thì next open <> thì tự động dowload
if isExist == False:
    print('--> Tải file weights....')
    download(weights_url, modelWeights)
    print('--> Tải hoàn tất!')


# -- đọc file --
classNames = []     
with open(classFiles, 'rt') as f:
    print('>> Đang đọc file class....', end = ' ')
    classNames = f.read().rstrip('\n').split('\n')
    print('Hoàn tất!')

# -- Cung cấp các tệp cấu hình và trọng số cho mô hình và tải mạng. -- model DNN
print('>> Đang kết nối mạng....', end='')
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print('Hoàn tất!')

# Đọc hình ảnh từ base64 và chuyển đổi thành định dạng blob để đưa vào mạng
def readImage(img_b64): # đọc hình ảnh và chuyển đổi nó thành blob
    
    # https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    encoded_data = img_b64.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    blob = cv2.dnn.blobFromImage(img, 1/255, (whTarget, whTarget), [0, 0, 0], 1, crop= False)

    return img, blob

# mạng nơ-ron để phát hiện đối tượng, có khả năng là với mô-đun DNN của OpenCV.
def findObject(outputs, img):
    hT, wT, cT = img.shape
    boundingBox = []
    classIds = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            
            if confidence > confThreshold:
                w, h = int(detection[2]*wT), int(detection[3]*hT)
                # center point
                x, y = int(detection[0]*wT - w/2), int(detection[1]*hT - h/2)
                boundingBox.append([x, y, w, h])
                classIds.append(classId)
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boundingBox, confidences, confThreshold,mnsThreshold)

    for i in indices:
        try:
            i = i[0]
        except:
            pass 

        box = boundingBox[i]
        x, y, w, h = box[0], box[1], box[2], box[3] 
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confidences[i]*100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# sử dụng js trong file
@eel.expose
def animalDetection(img_b64):
    print('>> Đã phát hiện ảnh....', end =' ')
    img, blob = readImage(img_b64)

    # blob kết nối tới mạng
    net.setInput(blob)

    # xác định lớp đầu ra
    layerNames = net.getLayerNames()

    try:
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    except:
        outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

    # lấy đầu ra từ ở đoạn trên
    outputs = net.forward(outputNames)

    findObject(outputs, img)

    # lưu hình ảnh đã phát hiện
    cv2.imwrite(detectedImgPath, img)
    print('Hoàn tất!')


# -- mở ứng dụng trên Desktop --
print('>> Bắt đầu chương trình!')
eel.start('index.html', size=(1010, 680))