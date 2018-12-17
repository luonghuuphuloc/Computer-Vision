import numpy as np
import cv2

'''
  =============================================================================================
              Binary encoding of data digits into EAN-13 Barcode section                 
  =============================================================================================


                              Structure of EAN-13                                    
       _______________________________________________________________________
      | First digit   | First group of 6 digits   | Last group of 6 digits    |
      |_______________|___________________________|___________________________|
      |       0       |           LLLLLL          |           RRRRRR          |
      |       1       |           LLGLGG          |           RRRRRR          |
      |       2       |           LLGGLG          |           RRRRRR          |   
      |       3       |           LLGGGL          |           RRRRRR          |
      |       4       |           LGLLGG          |           RRRRRR          |
      |       5       |           LGGLLG          |           RRRRRR          |
      |       6       |           LGGGLL          |           RRRRRR          |
      |       7       |           LGLGLG          |           RRRRRR          |   
      |       8       |           LGLGGL          |           RRRRRR          |
      |       9       |           LGGLGL          |           RRRRRR          |
      |_______________|___________________________|___________________________|


                          Encoding of the digits
       _______________________________________________________________________
      |   Digit   |       L-code      |       G-code      |       R-code      |
      |___________|___________________|___________________|___________________|
      |     0     |      0001101      |      0100111      |       1110010     |
      |     1     |      0011001      |      0110011      |       1100110     |
      |     2     |      0010011      |      0011011      |       1101100     |
      |     3     |      0111101      |      0100001      |       1000010     |
      |     4     |      0100011      |      0011101      |       1011100     |
      |     5     |      0110001      |      0111001      |       1001110     |
      |     6     |      0101111      |      0000101      |       1010000     |
      |     7     |      0111011      |      0010001      |       1000100     |
      |     8     |      0110111      |      0001001      |       1001000     |
      |     9     |      0001011      |      0010111      |       1110100     |
      |___________|___________________|___________________|___________________|
'''

L_R = { "3211": '0',
        "2221": '1',
        "2122": '2',
        "1411": '3',
        "1132": '4',
        "1231": '5',
        "1114": '6',
        "1312": '7',
        "1213": '8',
        "3112": '9'}

G = {  "1123": '0',
       "1222": '1',
       "2212": '2',
       "1141": '3',
       "2311": '4',
       "1321": '5',
       "4111": '6',
       "2131": '7',
       "3121": '8',
       "2113": '9'}

Region = { "LLLLLL": '0',
           "LLGLGG": '1',
           "LLGGLG": '2',
           "LLGGGL": '3',
           "LGLLGG": '4',
           "LGGLLG": '5',
           "LGGGLL": '6',
           "LGLGLG": '7',
           "LGLGGL": '8',
           "LGGLGL": '9'}

#   =============================================================================================
#           End of Binary encoding of data digits into EAN-13 Barcode section
#   =============================================================================================


#   =============================================================================================
#                                Barcode detecting section
#   =============================================================================================

'''
Detect Algorithm
1. First, we use sobel gradient on frame to make the barcode easier to detect
2.  
'''
def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    X = cv2.Sobel(gray, ddepth = cv2.CV_64F, dx = 1, dy = 0, ksize = -1)
    Y = cv2.Sobel(gray, ddepth = cv2.CV_64F, dx = 0, dy = 1, ksize = -1)

    grad_sobel = np.sqrt(X * X + Y * Y)

    cv2.normalize(grad_sobel, grad_sobel, 0, 3, cv2.NORM_MINMAX)

    blurred = cv2.blur(grad_sobel, (10, 10))
    minVal, maxVal, _, _ = cv2.minMaxLoc(blurred)
    mean, stddev = cv2.meanStdDev(blurred)

    _, thresh = cv2.threshold(blurred, mean + stddev, maxVal, cv2.THRESH_BINARY) 

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    opened = cv2.convertScaleAbs(opened)
    
    _, contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        x, y, w, h = 200, 200, 200, 100
    else:
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[0]
        x, y, w, h = cv2.boundingRect(contours)


    roi = frame[y : y + h, x : x + w]
    roi = cv2.resize(roi, (1200, 600), cv2.INTER_LINEAR)
    
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    roi = cv2.GaussianBlur(roi,(5,5),0)
    roi = cv2.equalizeHist(roi)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return roi



#   =============================================================================================
#                                   Barcode decoding section
#   =============================================================================================


def decode(roi):
    result = ""
    for thr in range(5, 100,3):
        _, binary = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY)
        
        #   Intensity of black and white pixels of binary image
        intensity = np.zeros(roi.shape[1] + 1)
        intensity[0 : roi.shape[1]] = binary[int(roi.shape[0] / 2), 0 : roi.shape[1]]
        intensity[roi.shape[1]] = 255
        left, right = 0, 0

        #   Find the first black pixel of intensity array
        for i in range(0, 1200):
            if (intensity[i] == 0):
                left = i
                break

        #   Find the last black pixel of intensity array
        for i in range(0, 1200):
            if (intensity[1199 - i] == 0):
                right = 1199 - i
                break

        count = 1
        num = []
        n = 0

        if left != 0 and right != 0: 
            for i in range(left + 1 , right + 2):
                if intensity[i - 1] == intensity[i]:
                    count += 1
                else:
                    num.append(count)
                    count = 1

        np.asarray(num, dtype = int)

        if len(num) != 59:
            continue
        else:

            black = (num[0] + num[2] + num[28] + num[30] + num[56] + num[58]) / 6
            white = (num[1] + num[27] + num[29] + num[31] + num[57]) / 5

            index = []
            for i in range(len(num)):
                if i % 2 == 0:
                    k = black
                else:
                    k = white   
                if (num[i] / k) - np.floor(num[i] / k) < 0.33:
                    index.append(int(round(num[i] / k)))
                else:
                    index.append(int(np.ceil(num[i] / k)))
            np.asarray(index, dtype = int)

            #   Delete detector bars
            index = np.delete(index, [0, 1, 2, 27, 28, 29, 30, 31, 56, 57, 58])
            
            check = ""
            for i in range(0, len(index), 4):
                key = np.array2string(index[i]) + np.array2string(index[i + 1]) + \
                         np.array2string(index[i + 2]) + np.array2string(index[i + 3])

                if key in L_R:
                    result += L_R[key]
                    check += 'L'
                elif key in G:
                    result += G[key]
                    check += 'G'

            if len(result) == 12:

                check = check[0 : 6]
                result = Region[check] + result
                break 
            else:
                result = ""

    return result