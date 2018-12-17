import numpy as np
import cv2
import math
import setting

def checkRatio(stateCount):
	totalFinderSize = 0
	for i in range(0, 5):
		count = stateCount[i]
		totalFinderSize += count
		if count == 0:
			return 0
	if totalFinderSize < 7:
		return 0
	moduleSize = math.ceil(totalFinderSize / 7.0)	
	maxVariance = moduleSize / 2
	retVal = ((abs(moduleSize - (stateCount[0])) < maxVariance) and \
		(abs(moduleSize - (stateCount[1])) < maxVariance) and \
		(abs(3 * moduleSize - (stateCount[2])) < 3 * maxVariance) and \
		(abs(moduleSize - (stateCount[3])) < maxVariance) and \
		(abs(moduleSize - (stateCount[4])) < maxVariance))
	
	return retVal

def centerFromEnd(stateCount, end):
	center = (end - stateCount[4] - stateCount[3]) - stateCount[2] / 2
	center = math.ceil(center)
	return center

def handlePossibleCenter(img, row, col):
	stateCountTotal = 0
	diff = [0, 0]
	for i in range (0, 5):
		stateCountTotal = stateCountTotal + setting.stateCount[i]
	centerCol = centerFromEnd(setting.stateCount, col)
	centerRow = crossCheckVertical(img, row, centerCol, setting.stateCount[2], stateCountTotal)
	
	if(centerRow == None):
		return 0
	if np.isnan(centerRow):
		return 0
	centerCol = crossCheckHorizontal(img, centerRow, centerCol, setting.stateCount[2], stateCountTotal)
	if(centerCol == None):
		return 0
	if np.isnan(centerCol):
		return 0
	validPattern = crossCheckDiagonal(img, centerRow, centerCol, setting.stateCount[2], stateCountTotal)
	if validPattern == 0:
		
		return 0
	ptNew = [centerCol, centerRow]
	
	newEstimatedModuleSize = stateCountTotal / 7.0
	found = 0
	idx = 0
	pt = setting.possibleCenters
	
	for index1 in range(len(pt)):
		
		diff[0] = pt[index1][0] - ptNew[0]
		diff[1] = pt[index1][1] - ptNew[1]
		dist = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
		if(dist < 10):
			pt[index1][0] = pt[index1][0] + ptNew[0]
			pt[index1][1] = pt[index1][1] + ptNew[1]
			pt[index1][0] /= float(2.0)
			pt[index1][1] /= float(2.0)
			
			setting.estimatedModuleSize[idx] = (setting.estimatedModuleSize[idx] + newEstimatedModuleSize) / float(2.0)
			found = 1
			break
		idx = idx + 1
	if(found == 0):
		setting.possibleCenters.append(ptNew)
		setting.estimatedModuleSize.append(newEstimatedModuleSize)
		
	return 0

def crossCheckVertical(img, startRow, centerCol, centralCount, stateCountTotal):
	maxRows, _ = img.shape
	
	crossCheckStateCount = [0, 0, 0, 0, 0]
	row = startRow
	
	while (row >= 0 and img[row, centerCol] < 128):
		crossCheckStateCount[2] = crossCheckStateCount[2] + 1
		row = row - 1
	if(row < 0):
		return 0
	while(row >= 0 and img[row, centerCol] >= 128 and crossCheckStateCount[1] < centralCount):
		crossCheckStateCount[1] = crossCheckStateCount[1] + 1
		row = row - 1
	if(row < 0 or crossCheckStateCount[1] >= centralCount):
		return 0
	while(row >= 0 and img[row, centerCol] < 128 and crossCheckStateCount[0] < centralCount):
		crossCheckStateCount[0] = crossCheckStateCount[0] + 1
		row = row - 1
	if(row < 0 or crossCheckStateCount[0] >= centralCount):
		return 0
	row = startRow + 1
	while(row < maxRows and img[row, centerCol] < 128):
		crossCheckStateCount[2] = crossCheckStateCount[2] + 1
		row = row + 1
	if(row == maxRows):
		return 
	while(row < maxRows and img[row, centerCol] >= 128 and crossCheckStateCount[3] < centralCount):
		crossCheckStateCount[3] = crossCheckStateCount[3] + 1
		row = row + 1
	if(row == maxRows or crossCheckStateCount[3] >= stateCountTotal):
		return 0
	while(row < maxRows and img[row, centerCol] < 128 and crossCheckStateCount[4] < centralCount):
		crossCheckStateCount[4] = crossCheckStateCount[4] + 1
		row = row + 1
	if(row == maxRows or crossCheckStateCount[4] >= centralCount):
		return 0
	crossCheckStateCountTotal = 0
	for i in range(0,5):
		crossCheckStateCountTotal = crossCheckStateCountTotal + crossCheckStateCount[i]
	if(5 * abs(crossCheckStateCountTotal - stateCountTotal) >= 2 * stateCountTotal):
		return 0
	center  = centerFromEnd(crossCheckStateCount, row)
	if (checkRatio(crossCheckStateCount)):
		return center
	else:
		return 0

def crossCheckHorizontal(img, centerRow, startCol, centerCount, stateCountTotal):
	_,maxCols = img.shape
	HorizontalstateCount = [0, 0, 0, 0, 0]
	col = startCol	
	
	while (col >= 0 and img[centerRow, col] < 128):
		HorizontalstateCount[2] += 1
		col -= 1
	if (col < 0):
		return 0
	
	while (col >= 0 and img[centerRow, col] >= 128 and HorizontalstateCount[1] < centerCount):
		HorizontalstateCount[1] += 1
		col -= 1
	if (col < 0 or HorizontalstateCount[1] == centerCount):
		return 0

	while (col >= 0 and img[centerRow, col] < 128 and HorizontalstateCount[0] < centerCount):
		HorizontalstateCount[0] += 1
		col -= 1
	if (col < 0 or HorizontalstateCount[0] == centerCount):
		return 0

	col = startCol + 1
	while (col < maxCols and img[centerRow, col] < 128):
		HorizontalstateCount[2] += 1
		col += 1
	if (col == maxCols): 
		return 0
	

	while (col < maxCols and img[centerRow, col] >= 128 and HorizontalstateCount[3] < centerCount): 
		HorizontalstateCount[3] += 1
		col += 1
	if (col == maxCols or HorizontalstateCount[3] == centerCount): 
		return 0
	

	while (col < maxCols and img[centerRow, col] < 128 and HorizontalstateCount[4] < centerCount): 
		HorizontalstateCount[4] += 1
		col += 1
	if (col == maxCols or HorizontalstateCount[4] == centerCount):
		return 0
	
	newStateCountTotal = 0
	for  i in range(0,5):
		newStateCountTotal += HorizontalstateCount[i]
	

	if (5 * abs(stateCountTotal - newStateCountTotal) >= stateCountTotal): 
		return 0
	

	if(checkRatio(HorizontalstateCount)):
		return centerFromEnd(HorizontalstateCount, col) 
	else:
		return 0


def crossCheckDiagonal(img, centerRow, centerCol, maxCount, stateCountTotal):
	DiagonalstateCount = [0, 0, 0, 0, 0] 
	
	i = 0
	while (centerRow >= i and centerCol >= i and img[centerRow - i, centerCol - i] < 128): 
		DiagonalstateCount[2] += 1
		i += 1
	
	if (centerRow < i or centerCol < i):
		return 0
	

	while (centerRow >= i and centerCol >= i and img[centerRow - i, centerCol - i] >= 128 and DiagonalstateCount[1] <= maxCount): 
		DiagonalstateCount[1] += 1
		i += 1
	
	if (centerRow < i or centerCol < i or DiagonalstateCount[1] > maxCount): 
		return 0
	

	while (centerRow >= i and centerCol >= i and img[centerRow - i, centerCol - i] < 128 and DiagonalstateCount[0] <= maxCount): 
		DiagonalstateCount[0] += 1
		i += 1
	
	if (DiagonalstateCount[0] > maxCount): 
		return 0
	
	maxRows,maxCols = img.shape
	
	i = 1
	while ((centerRow + i) < maxRows and (centerCol + i) < maxCols and img[centerRow + i, centerCol + i] < 128): 
		DiagonalstateCount[2] += 1
		i += 1
	
	if ((centerRow + i) >= maxRows or (centerCol + i) >= maxCols): 
		return 0
	

	while ((centerRow + i) < maxRows and (centerCol + i) < maxCols and img[centerRow + i, centerCol + i] >= 128 and DiagonalstateCount[3] < maxCount): 
		DiagonalstateCount[3] += 1
		i += 1
	
	if ((centerRow + i) >= maxRows or (centerCol + i) >= maxCols or DiagonalstateCount[3] > maxCount): 
		return 0
	

	while ((centerRow + i) < maxRows and (centerCol + i) < maxCols and img[centerRow + i, centerCol + i] < 128 and DiagonalstateCount[4] < maxCount): 
		DiagonalstateCount[4] += 1
		i += 1
	
	if ((centerRow + i) >= maxRows or (centerCol + i) >= maxCols or DiagonalstateCount[4] > maxCount): 
		return 0
	
	newStateCountTotal = 0
	for j in range(0,5): 
		newStateCountTotal += DiagonalstateCount[j]
	

	if((abs(stateCountTotal - newStateCountTotal) < 2 * stateCountTotal) > 0 and  checkRatio(DiagonalstateCount)):
		return 1
	else:
		return 0

def drawFinders(img):
	if (len(setting.possibleCenters) == 0): 
		return
	
	for i in range(len(setting.possibleCenters)):
		pt = setting.possibleCenters[i]
		diff = setting.estimatedModuleSize[i] * 3.5

		pt1 = [math.ceil(pt[0] - diff), math.ceil(pt[1] - diff)]
		pt2 = [math.ceil(pt[0] + diff), math.ceil(pt[1] + diff)]
		cv2.rectangle(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (255, 0, 0), 1)

def computeDimension(tl, tr, bl, moduleSize):
	diff_top = [tr[0] - tl[0], tr[1] - tl[1]]
	diff_left = [tl[0] - bl[0], tl[1] - bl[1]]
	global dist_top
	global dist_left
	dist_top = np.sqrt(diff_top[0] ** 2 + diff_top[1] ** 2)
	dist_left = np.sqrt(diff_left[0] ** 2 + diff_left[1] ** 2)
	width = round(dist_top / moduleSize)
	height = round(dist_left / moduleSize)
	dimension = int((width + height) / 2) + 7
	
	if((dimension % 4) == 0):
		dimension += 1
		return dimension
	if((dimension % 4) == 1):
		return dimension
	if((dimension % 4) == 2):
		dimension -= 1
		return dimension
	if((dimension % 4) == 3):
		dimension -= 2
		return dimension

def findAlignmentMarker(img):
	if(len(setting.possibleCenters) != 3):
		return 0
	temp = []
	
	if(setting.possibleCenters[0][0] > setting.possibleCenters[1][0]):
		temp = setting.possibleCenters[0]
		setting.possibleCenters[0] = setting.possibleCenters[1]
		setting.possibleCenters[1] = temp

	ptTopLeft = setting.possibleCenters[0]
	ptTopRight = setting.possibleCenters[1]
	ptBottomLeft = setting.possibleCenters[2]
	moduleSize = (setting.estimatedModuleSize[0] + setting.estimatedModuleSize[1] + setting.estimatedModuleSize[2]) / float(3.0)
	global dimension
	dimension = computeDimension(ptTopLeft, ptTopRight, ptBottomLeft, moduleSize)
	
	if(dimension == 21 or 1):
        
		ptBottomRight = [ptTopRight[0] - ptTopLeft[0] + ptBottomLeft[0],ptBottomLeft[1] - ptTopLeft[1] + ptTopRight[1]]
		setting.possibleCenters.append(ptBottomRight)
		return 1;
	return 0

def getTransformedMarker(img):
	moduleSize = (setting.estimatedModuleSize[0] + setting.estimatedModuleSize[1] + setting.estimatedModuleSize[2]) / float(3.0)
	x1 = [setting.possibleCenters[0][0] - 3 * moduleSize,setting.possibleCenters[0][1] - 3 * moduleSize]
	x2 = [setting.possibleCenters[1][0] + 3 * moduleSize,setting.possibleCenters[1][1] - 3 * moduleSize]
	x3 = [setting.possibleCenters[2][0] - 3 * moduleSize,setting.possibleCenters[2][1] + 3 * moduleSize]
	x4 = [setting.possibleCenters[3][0] + 3 * moduleSize,setting.possibleCenters[3][1] + 3 * moduleSize]

	pt1 = [float(3), float(3)]
	pt2 = [dist_top + 2 * moduleSize  - float(3), float(3)]
	pt3 = [float(3),dist_left + 2 * moduleSize - float(3)]
	pt4 = [dist_top + 2 * moduleSize  - float(3),dist_left + 2 * moduleSize  - float(3)]
	src = np.asarray([pt1, pt2, pt3, pt4], dtype = "float32")
	dts = np.asarray([x1, x2, x3, x4], dtype = "float32")

	transform = cv2.getPerspectiveTransform(dts, src)
	marker = cv2.warpPerspective(img, transform, (int(dist_top + 2.0 * moduleSize ), int(dist_left + 2.0 * moduleSize)))

	marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
	marker = cv2.adaptiveThreshold(marker, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 0)
	#Resize the marker
	marker = cv2.resize(marker, (dimension * 10,dimension * 10), cv2.INTER_LINEAR)
	#Chose value is the bit center
	global data
	data = np.zeros(shape = (dimension, dimension))
	for x in range(5, dimension * 10, 10):
		for y in range(5, dimension * 10, 10):
			if(float(marker[x, y]) > 10):
				data[int(x / 10), int(y / 10)] = 1
			else:
				data[int(x / 10), int(y / 10)] = 0
	
	cv2.imshow('marker', marker)
	cv2.moveWindow('marker', 1200, 0)
	
def get_dead_zone():
	constant_zones = [
        (0, 0, 8, 8),  # top left position + format-info
        (size - 8, 0, size - 1, 8),  # top right position + format-info
        (0, size - 8, 7, size - 1),  # bottom left position
        (8, size - 7, 8, size - 1),  # bottom left format info
        (8, 6, size - 9, 6),  # top timing array
        (6, 8, 6, size - 9)  # left timing array
    ]
    #return constant_zones
def get_format_info():
	fm1_l = data[8, 0 : 6]
	fm1_l = np.append(fm1_l, data[8, 7])
	fm1_h = data[0 : 6, 8]
	fm1_h = np.append(fm1_h, data[7, 8])
	
	mask1 = np.asarray([0, 0, 0])
	mask2 = np.asarray([0, 0, 1])
	mask3 = np.asarray([0, 1, 0])
	mask4 = np.asarray([0, 1, 1])
	mask5 = np.asarray([1, 0, 0])
	mask6 = np.asarray([1, 0, 1])
	mask7 = np.asarray([1, 1, 0])
	mask8 = np.asarray([1, 1, 1])

	format_mask = fm1_l[2 : 5]
	global format_num
	if(format_mask == mask1).all():
		#mask: j%3 = 0
		format_num = "000"

	if(format_mask == mask2).all():
		#mask: (i+j)%3 = 0
		format_num = "001"

	if(format_mask == mask3).all():
		#mask: (i+j)%2 = 0
		format_num = "010"

	if(format_mask == mask4).all():
		#mask: i%2 = 0
		format_num = "011"

	if(format_mask == mask5).all():
		#mask: ((i*j)%3 +i*j)%2 = 0
		format_num = "100"

	if(format_mask == mask6).all():
		#mask: ((i*j)%3 +i+j)%2 = 0
		format_num = "101"

	if(format_mask == mask7).all():
		#mask: (i/2+j/2)%2 = 0
		format_num = "110"

	if(format_mask == mask8).all():
		#mask: (i*j)%2 +(i*j)%3 = 0
		format_num = "111"
	
def scan_mask():
	mask_done_flag = 0
	if(format_num == "000"):
		for i in range(0, dimension, 1):
			for j in range(0, dimension, 1):
				if j % 3 == 0:
					if data[i, j] == 0:
						data[i, j] = 1
					else:
						data[i, j] = 0
		mask_done_flag = 1

	if(format_num == "001"):
		for i in range(0, dimension, 1):
			for j in range(0, dimension, 1):
				if (i + j) % 3 == 0:
					if data[i, j] == 0:
						data[i, j] = 1
					else:
						data[i, j] = 0
		mask_done_flag = 1
	
	if(format_num == "010"):
		for i in range(0, dimension, 1):
			for j in range(0, dimension, 1):
				if (i + j) % 2 == 0:
					if data[i, j] == 0:
						data[i, j] = 1
					else:
						data[i, j] = 0
		mask_done_flag = 1	

	if(format_num == "011"):
		for i in range(0, dimension, 1):
			for j in range(0, dimension, 1):
				if(i % 2 == 0):
					if data[i, j] == 0:
						data[i, j] = 1
					else:
						data[i, j] = 0
		mask_done_flag = 1	

	if(format_num == "100"):
		for i in range(0, dimension, 1):
			for j in range(0, dimension, 1):
				if ((i * j) % 3 + i * j) % 2 == 0:
					if data[i,j] == 0:
						data[i,j] = 1
					else:
						data[i,j] = 0
		mask_done_flag = 1	

	if(format_num == "101"):
		for i in range(0, dimension, 1):
			for j in range(0, dimension, 1):
				if((i * j) % 3 + i + j) % 2 == 0:
					if data[i, j] == 0:
						data[i, j] = 1
					else:
						data[i, j] = 0
		mask_done_flag = 1	

	if(format_num == "110"):
		for i in range(0, dimension, 1):
			for j in range(0, dimension, 1):
				if((i / 2 + j / 3) % 2 == 0):
					if data[i, j] == 0:
						data[i, j] = 1
					else:
						data[i, j] = 0
		mask_done_flag = 1	

	if(format_num == "111"):
		for i in range(0, dimension, 1):
			for j in range(0, dimension, 1):
				if (i * j) % 2 + (i * j) % 3 == 0:
					if data[i, j] == 0:
						data[i, j] = 1
					else:
						data[i, j] = 0
		mask_done_flag = 1
	#reverse bit because in QRCode: dot Black is 1, and dot White is 0 
	if(mask_done_flag == 1):
		for i in range(0, dimension):
			for j in range(0, dimension):
				if data[i, j] == 0:
					data[i, j] = 1
				else:
					data[i, j] = 0
		

	#Split zig zag Data array
	global real_data
	real_data = []
	k = 1
	h = 0
	for i in range(0, 5):
		for j in range(0, 12):
			real_data = np.append(real_data, data[(20 - j) * k + (9 + j) * h, 20 - 2 * i])
			real_data = np.append(real_data, data[(20 - j) * k + (9 + j) * h, 19 - 2 * i])
		if(k == 1 or h == 0):
			k = 0
			h = 1
		else:
			k = 1
			h = 0
	k = 1
	h = 0
	for i in range(0, 2):
		for j in range(0, 9):
			if((8 - j) * k + (0 + j) * h != 6):
				real_data = np.append(real_data, data[(8 - j) * k + (0 + j) * h, 12 - 2 * i])
				real_data = np.append(real_data, data[(8 - j) * k + (0 + j) * h, 11 - 2 * i])
		k = 0
		h = 1
	
	#Split encoding mode from Data array above
	global enc_mode
	enc_mode = real_data[0 : 4]
	
	#Split message length from Data array above
	global msg_length
	msg_length = real_data[4 : 12]

def decode_Qrcode():
	global end_flag
	end_flag = 0
	msg_final = []
	byte_enc = np.asarray([0, 1, 0, 0])
	#Nếu Encoding mode là kiểu Byte
	if(enc_mode == byte_enc).all():
		length_num = cal_byte(msg_length)
		#print("Byte_encoding with msg_length = ",length_num)
		for i in range(0,length_num):
			msg_temp = real_data[12 + 8 * i : 20 + 8 * i]
			msg_temp = cal_byte(msg_temp)
			msg_final = np.append(msg_final, chr(int(msg_temp)))
		a = ''.join(msg_final)
		
		end_flag = 1
		return a
	#If Encoding mode is another type or just we detect incorrect
	else:
		end_flag = 0

			

def cal_byte(num):
	if (len(num) != 8):
		return 0
	num_out = int(num[0] * 128 + num[1] * 64 + num[2] * 32 + num[3] * 16 \
	+ num[4] * 8 + num[5] * 4 + num[6] * 2 + num[7])
	return	num_out

