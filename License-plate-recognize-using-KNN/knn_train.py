import numpy as np
import cv2
import os 
#import imp
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.metrics import accuracy_score


# Sử dụng KNN của thư viện Scikit-learn

# Data gồm các kí tự số từ 0-9, chữ từ A-Z
# mỗi kí tự gồm 20 hình với độ nghiêng khác nhau

# Data này do nhóm tự tạo dựa trên hình chứa font 
# trong file BTVN08


path = 'Data'
data = data_test = np.empty((0, 20*50))
labels = []
lables_test = []


# Lần lượt load từng image trong mỗi folder
# để trích đặc trưng cho mạng KNN 

# Data chia làm 2 phần: 10 kí tự để train
# và 10 kí tự dùng để test

# Nhãn của mỗi đặc trưng tương ứng là tên folder chứa ảnh

for label in os.listdir(path):
    imgPath  = path  +'/' + label
    i = 0 
    for image in os.listdir(imgPath):
        img = cv2.imread(path  +'/' + label + '/' + image,1)
        img = cv2.resize(img,(20,50),cv2.INTER_LINEAR)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
        gray = cv2.medianBlur(gray,5)
        ret, binary = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV) 

        # Chuyển hình thành vector 1x1000 làm đặc trưng
        H = binary.reshape((1,20*50))
        
        if i <10:
            data = np.append(data,H,0)
            labels.append(ord(label))
        else:
            data_test = np.append(data_test,H,0)
            lables_test.append(ord(label))
        i += 1


# Reshape lại label theo cột để tương ứng với đặc trưng

data = np.asarray(data,np.float32)
labels = np.asarray(labels,np.float32)

data_test = np.asarray(data_test,np.float32)
lables_test = np.asarray(lables_test,np.float32)


# Huấn luyện mạng, tính toán độ chính xác của mạng
knn = KNN(n_neighbors = 3) 
knn.fit(data,labels)
label_predict = knn.predict(data_test)

# Tính toán độ chính xác
print('Accuracy:', accuracy_score(lables_test,label_predict)*100)

# Lưu lại giá trị mạng KNN để sử dụng cho chương trình main
joblib.dump(knn, 'knn.pkl') 


