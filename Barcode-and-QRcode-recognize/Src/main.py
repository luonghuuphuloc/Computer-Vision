from tkinter import *
from tkinter.messagebox import *
import numpy as np
import cv2
# from PIL import ImageTk,Image 

#	_____________________________________________________________________________________________________
    
def main_decoder():	
	
	if v1.get() == 1 and v2.get() == 0:

	#	=================================================================================================
	#										BARCODE SECTION
	#	=================================================================================================		
		
		import Barcode	
		import os
		
	

		while(1):
		    _, frame = cap.read()

		    #	Barcode detecting
		    roi = Barcode.detect(frame)		    
		    cv2.imshow('Camera', frame)
		    cv2.moveWindow('Camera',562,0)
		    
		    #	Barcode decoding
		    result = Barcode.decode(roi)
		    
		    if len(result) == 13:
		        
		        print('Barcode:', result)
		        code.set(result)
		        
		        if len(code_txt.get("1.0", "end-1c")) != 0:
		        	code_txt.delete("1.0", END)
		        
		        code_txt.insert(END, code.get())


		        # code_ph = PhotoImage(file = "code_img.pgm")
		        # lb_5 = Label(root, image = code_ph)
		        # lb_5.place(x = 0, y = 0)
		        
		        break
		        
		    if cv2.waitKey(25) & 0xFF == 27:
		    	break
		cv2.destroyAllWindows()
	#	=================================================================================================
	#										END OF BARCODE SECTION
	#	=================================================================================================

	

	
	if v1.get() == 0 and v2.get() == 1:

	#	=================================================================================================
	#											QR-CODE SECTION
	#	=================================================================================================
			
		import math
		import QRCode
		import setting

		while(1):
			_, img = cap.read()
			cv2.imshow('Camera', img)
			cv2.moveWindow('Camera',562,0)

			img_BW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)			
			img_BW = cv2.adaptiveThreshold(img_BW, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 0)

			rows, cols = img_BW.shape
			setting.possibleCenters.clear()
			setting.estimatedModuleSize.clear()
			skipRows = 3

			for row in range(skipRows-1, rows, skipRows):
				setting.stateCount[0] = 0
				setting.stateCount[1] = 0
				setting.stateCount[2] = 0
				setting.stateCount[3] = 0
				setting.stateCount[4] = 0
				setting.currentState = 0

				for col in range(0, cols):
					
					if img_BW[row, col] <128:
						
						if (setting.currentState & 0x1) == 1:
							setting.currentState += 1
						setting.stateCount[setting.currentState] += 1
					
					else:
						
						if (setting.currentState & 0x1) == 1:
							setting.stateCount[setting.currentState] += 1
						
						else:
							
							if setting.currentState == 4:	
								
								if (QRCode.checkRatio(setting.stateCount)):
									confirmed = QRCode.handlePossibleCenter(img_BW, row, col)
								
								else:
									setting.currentState = 3
									setting.stateCount[0] = setting.stateCount[2]
									setting.stateCount[1] = setting.stateCount[3]
									setting.stateCount[2] = setting.stateCount[4]
									setting.stateCount[3] = 1
									setting.stateCount[4] = 0
									
									continue

								setting.currentState = 0
								setting.stateCount[0] = 0
								setting.stateCount[1] = 0
								setting.stateCount[2] = 0
								setting.stateCount[3] = 0
								setting.stateCount[4] = 0
							
							else:
								setting.currentState += 1
								setting.stateCount[setting.currentState] += 1

			if (len(setting.possibleCenters) > 0):
				QRCode.drawFinders(img);
				QRCode.findAlignmentMarker(img)
				
				if(len(setting.possibleCenters) == 4):
					QRCode.getTransformedMarker(img)
					QRCode.get_format_info()
					QRCode.scan_mask()
					message = QRCode.decode_Qrcode()
					
					if(QRCode.end_flag == 1):
						print('QRcode', message)
						code.set(message)
						
						if len(code_txt.get("1.0", "end-1c")) != 0:
							code_txt.delete("1.0", END)
						
						code_txt.insert(END, code.get() )

						break
			
			if cv2.waitKey(25) & 0xFF == 27:
				break
		
		cv2.destroyAllWindows()						

	#	=================================================================================================
	#									END OF QR-CODE SECTION
	#	=================================================================================================



	if (v1.get() == 0 and v2.get() == 0) or (v1.get() == 1 and v2.get() == 1):
		showinfo('Notice', 'Which type of code do you want to decode?')
#	_____________________________________________________________________________________________________



#	_____________________________________________________________________________________________________

def close():	
	cap.release()
	quit()
#	_____________________________________________________________________________________________________


#   =====================================================================================================
#                              			GRAPHICAL USER INTERFACE    
#   =====================================================================================================

#	To create graphic user interface, we use Tkinter module.

if __name__ == "__main__":

	#   Capture Video. Because the quality of the image captured from the webcam of our laptop is  quite 
	#   low, so in this project we use iVCam which is set up on laptop and an iphone to stream the video 
	#   taken by the iphone. Both the laptop and the iphone are required  to  have same network and  the
	#   iVCam software

	cap = cv2.VideoCapture(1)	
	
	root = Tk()
	root.title("Barcode and QR-code Decoder")
	root.geometry("560x480+0+0") 
	root.configure(background = "light blue")
	

	logo1 = PhotoImage(file = "DHBK_logo.ppm")
	lb_1 = Label(root, image = logo1)
	lb_1.place(x = 0, y = 0)

	v1 = IntVar()
	v2 = IntVar()
	lb_3 = Label(root, text = """Choose type of code:""", font = "Arial 11 bold", 
							justify = LEFT, padx = 20, background ="light blue")
	lb_3.place(x = 0, y = 160)
	
	
	bar_chb = Checkbutton(root, text = "Barcode", font = "Arial 11 normal", 
							padx = 20, variable = v1, background ="light blue")
	bar_chb.place(x = 240, y = 160)
	
	qr_chb = Checkbutton(root, text = "QR-code", font = "Arial 11 normal", 
							padx = 20, variable = v2, background ="light blue")
	qr_chb.place(x = 410, y = 160)

	
	quit_bt = Button(root, text = "QUIT", font = "Arial 11 bold", fg = "red", 
							height = 3, width = 15 , command = close)
	quit_bt.place(x = 393, y = 400)
	
	start_bt = Button(root, text = "START", font = "Arial 11 bold", fg = "blue", 
							height = 3, width = 15, command = main_decoder)
	start_bt.place(x = 20, y = 400)
	
	
	lb_4 = Label(root, text = """CODE: """, font = "Arial 11 bold", 
							justify = LEFT, padx = 20, background ="light blue")
	lb_4.place(x = 0, y = 210)

	code = StringVar()
	code_txt = Text(root, height = 1, width = 25)
	code_txt.configure(font = ('Verdana 20 bold'), padx = 20, pady = 50)
	code_txt.place(x = 20, y = 240)

	root.mainloop()

#   ========================================================================================================
#                              			END OF GRAPHIC USER INTERFACE    
#   ========================================================================================================