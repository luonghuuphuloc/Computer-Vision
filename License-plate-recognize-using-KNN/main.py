import numpy as np
import cv2
#import imp
from sklearn.externals import joblib

# Sử dụng công cụ joblib trong thư viện scikit-learn
# để load bộ phân loại knn đã training từ trước
# Chú ý đường dẫn đến file knn.pkl
# Trong project này, file knn.pkl nằm trong cùng 1 thư mục với file main
# nếu không, phải thay đổi đường dẫn đến folder chứa file này
knn = joblib.load('knn.pkl')

# Tên file hoặc đường dẫn đến file muốn nhận dạng
image_name = 'img5.png'

# Hàm con sắp xếp 4 điểm theo thứ tự: 
# top-left, top-right, bottom-right, bottom-left
def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	# top-left có tổng 2 toạ độ (x,y) nhỏ nhất
	# ngược lại bottom-right có tổng lớn nhất
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# top-right có hiệu giữa 2 toạ độ (x,y) nhỏ nhất
	# ngược lại, bottom-left có hiệu lớn nhất
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


# Hàm con biến đổi hình học
# dùng để đưa biển số xe bị nghiêng về lại đối diện người nhìn
def four_point_transform(image, pts):
	
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# Tính chiều rộng của ảnh sau khi biến đổi
	# Chiều rộng bằng maximum của khoảng cách giữa (bottom-left, bottom-right) và (top-left, top-right)
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# Tính chiều cao của ảnh sau khi biến đổi
	# Chiều cao bằng maximum của khoảng cách giữa (top-right, bottom-right) và (top-left, bottom-left)
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# Tính ma trận biến đổi hình học
	# sử dụng ma trận này để đưa hình bị nghiêng về lại thẳng, 
	# đối diện người nhìn
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	return warped


# Hàm con dùng để xác định vị trí của biển số,
# sau đó biến đổi hình học nếu biển số bị nghiêng,...
# Kết quả của hàm trả về là ma trận ảnh chứa biển số 
# và số lượng biển số trong hình
def plate_detect(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray,9,25,10)
	ret, binary = cv2.threshold(gray,157,255,cv2.THRESH_BINARY)
	i = 0
	thresh = np.zeros((10,140,190),np.uint8)

	_, contours, hierachy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	# Sắp xếp contour theo thứ tự trái sang phải
	sorted_ctrs = sorted(contours, key=lambda contours: cv2.boundingRect(contours)[0])

	for contour in sorted_ctrs:
		if cv2.contourArea(contour) > 500 and cv2.contourArea(contour) < 35000:
			(x,y,w,h) = cv2.boundingRect(contour)
			epsilon = 0.1*cv2.arcLength(contour,True)
			approx = cv2.approxPolyDP(contour,epsilon,True)
			if len(approx) == 4:
				approx = approx.reshape((4,2))
				warped = four_point_transform(img,approx)
				warped = cv2.resize(warped,(190,140),cv2.INTER_CUBIC)
				warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
				ret, binary = cv2.threshold(warped,155,255,cv2.THRESH_BINARY_INV)
				thresh[i] = binary
				i += 1
	
	return thresh, i



# Chương trình chính:


img = cv2.imread(image_name,1)

# Khởi tạo ma trận chứa hình biển số và số lượng
plates = np.zeros((10,140,190),np.uint8)

# Xác định biển số và số lượng biển số:
plates, n = plate_detect(img)

# Với mỗi biển số, sắp xếp các kí tự chữ,số
# theo thứ tự từ trái qua phải,
# hàng trên trước rồi đến hàng dưới
for i in range(0,n):
	binary = plates[i]
	cv2.imshow('binary'+np.str(i),binary)
	_, contours, hierachy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	cnt1 = []
	cnt2 = []
	for contour in contours:
		if cv2.contourArea(contour) > 350:
			if cv2.boundingRect(contour)[1] < 50:
				cnt1.append(contour)
			else:
				cnt2.append(contour)

	# Sắp xếp các kí tự chữ, số theo thứ tự
	sorted_ctrs = sorted(cnt1, key=lambda cnt1: cv2.boundingRect(cnt1)[0]) + sorted(cnt2, key=lambda cnt2: cv2.boundingRect(cnt2)[0])

	
	# Sau khi sắp xếp, lần lượt xác định từng kí tự
	# dựa trên mạng knn trước đó
	string = ""
	for contour in sorted_ctrs:
		(x,y,w,h) = cv2.boundingRect(contour)
		digit = binary[y:y+h,x:x+w]
		digit_resize = cv2.resize(digit,(20,50),cv2.INTER_LINEAR)
		
		# Trích đặc trưng của mỗi hình
		feature = digit_resize.reshape((1,20*50))
		feature = np.float32(feature)

		# Sử dụng mạng knn để tìm kí tự,
		# kết quả lưu lại vào biến string
		result = knn.predict(feature)
		strCurrentChar = str(chr(int(result))) 
		string += strCurrentChar	
	print('Result:',string)

	# Ghi kết quả lên hình tương ứng
	cv2.putText(img, string, (0+275*i,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
cv2.imshow('img',img)

cv2.waitKey(0)

