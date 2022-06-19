#!/bin/python3
# -*- coding: utf-8 -*-
# Coded By Kuduxaaa

import os, sys, easyocr, imutils, cv2, time, threading, sqlite3
import numpy as np

from matplotlib import pyplot as plt


PLATE = None
plateValue = None


class BoomBarrierController:
	def __init__(self):
		self.currentStatus = 'closed'


	def open(self):
		if self.getCurrentStatus() == 'closed':
			self.currentStatus = 'open'
			print('open')
			# შლაგბაუნის გაღების შემდეგ ავტომატურად დაკეტვის ალორითმია დასაწრერი აქ...

	def close(self):
		if self.currentStatus == 'open':
			self.currentStatus = 'closed'
			print('closed') 


	def isOpen(self):
		return False


	def getCurrentStatus(self):
		return self.currentStatus


	def startBeep(self, duration):
		print('Starting beep...')


class Database:
	def __init__(self):
		self.database = 'data/database.db'
		self.connection = sqlite3.connect(self.database)
		self.cursor = self.connection.cursor()


	def __del__(self):
		self.connection.close()


	def isAllowed(self, plateNumber):
		self.cursor.execute(
			'SELECT * FROM cars WHERE plate_number = ?', 
			(plateNumber,)
		)

		data = self.cursor.fetchone()

		if data is not None and len(data) > 0:
			if data[2] == 1:
				return True

		return False


class thread_with_trace(threading.Thread):
	def __init__(self, *args, **keywords):
		threading.Thread.__init__(self, *args, **keywords)
		self.killed = False


	def start(self):
		self.__run_backup = self.run
		self.run = self.__run     
		threading.Thread.start(self)


	def __run(self):
		sys.settrace(self.globaltrace)
		self.__run_backup()
		self.run = self.__run_backup



	def globaltrace(self, frame, event, arg):
		if event == 'call':
			return self.localtrace
		else:
			return None


	def localtrace(self, frame, event, arg):
		if self.killed:
			if event == 'line':
				raise SystemExit()
		
		return self.localtrace
 

	def kill(self):
		self.killed = True





class LPR:
	def __init__(self):
		self.version = 1.0
		self.db = Database()
		self.barrier = BoomBarrierController()


	def __repr__(self):
		return f'<LPR v{self.version}>'


	def showImage(self, image, title='Result', mode='matplotlib'):
		if mode == 'cv2':
			cv2.imshow(title, image)
			cv2.waitKey(9)
			cv2.destroyAllWindows()

		plt.imshow(
			cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		)

		plt.show()



	def imageToGray(self, image):
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



	def imageToEdge(self, image):
		bilateral 	= cv2.bilateralFilter(image, 11, 17, 17) # Noise reduction
		edged 	 	= cv2.Canny(bilateral, 30, 200) # Edge Detection

		return edged



	def findContours(self, image):
		points = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(points)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
		location = None

		for contour in contours:
			approx = cv2.approxPolyDP(contour, 10, True)
			if len(approx) == 4:
				location = approx
				break


		return location



	def cropPlateNumber(self, mask, gray):
		x, y = np.where(mask == 255)
		x1, y1 = (np.min(x), np.min(y)) # უკიდურესი ზედა კუთხეები
		x2, y2 = (np.max(x), np.max(y)) # უკიდურესი ქვედა კუთხეები

		return gray[x1:x2 + 1, y1:y2 + 1]




	def OpticalCharacterRecognition(self, image):
		reader = easyocr.Reader(['en'])
		result = reader.readtext(image)

		if len(result) > 0:
			return str(result[0][-2]).upper().replace('-', '').strip()

		return None



	def getImageWithPlateNumbers(self, image, approx, value):
		result = cv2.putText(
			image,
			text = value,
			org = (
				approx[0][0][0],
				approx[1][0][1] + 60
			),

			fontFace = cv2.FONT_HERSHEY_SIMPLEX,
			fontScale = 1,
			color = (0, 255, 0),
			thickness = 2,
			lineType = cv2.LINE_AA
		)

		return cv2.rectangle(image, 
			tuple(approx[0][0]),
			tuple(approx[2][0]),
			(0, 255, 0), 3)



	def recognitePlateNumber(self, filename, showImage = False):
		if not os.path.exists(filename):
			sys.exit(f'[-] File {filename} does not found')

		image = cv2.imread(filename)
		image = imutils.resize(image, 600, 480)

		gray = self.imageToGray(image)
		edged = self.imageToEdge(gray)

		approxContours = self.findContours(edged)
		mask = np.zeros(gray.shape, np.uint8)
		
		plate = cv2.drawContours(mask, [approxContours], 0, 255, -1)
		plate = cv2.bitwise_and(image, image, mask=mask)
		plate = self.cropPlateNumber(mask, gray)
		plateValue = self.OpticalCharacterRecognition(plate)

		if plateValue is None:
			return False

		if showImage:
			imageWithPlate = self.getImageWithPlateNumbers(image, approxContours, plateValue)
			self.showImage(imageWithPlate)

		return plateValue


	def OpticalCharacterRecognitionForVideo(self):
		global plateValue
		while True:
			if PLATE is not None:
				reader = easyocr.Reader(['en'])
				result = reader.readtext(PLATE)

				if len(result) > 0:
					plateValue = str(result[0][-2]).upper().replace('-', '').strip()

				else:
					plateValue = None


	def videoCapturing(self):
		global PLATE, plateValue
		capture = cv2.VideoCapture(0)
		opticalThread = thread_with_trace(target=self.OpticalCharacterRecognitionForVideo)
		opticalThread.start()

		while True:
			ret, image = capture.read()
			image = imutils.resize(image, 600, 480)

			gray = self.imageToGray(image)
			edged = self.imageToEdge(gray)

			mask = np.zeros(gray.shape, np.uint8)
			approxContours = self.findContours(edged)
			if approxContours is not None:
				plate = cv2.drawContours(mask, [approxContours], 0, 255, -1)
				plate = cv2.bitwise_and(image, image, mask=mask)
				plate = self.cropPlateNumber(mask, gray)

				if plateValue and plateValue is not None:
					if self.barrier.getCurrentStatus() != 'open':
						allowed = self.db.isAllowed(plateValue)

						if allowed:
							# თუ მანქანის ნომერი იპოვა ბაზაში
							# და მისი შესვლა ნებადართულია

							self.barrier.open()

						else:
							# თუ მანქანას არ აქვს შესვლის უფლება
							print('Access Denaid!')


					self.getImageWithPlateNumbers(image, approxContours, plateValue)

				PLATE = plate
			
			cv2.imshow('Video Capturing', image)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break


		if opticalThread.is_alive():
			opticalThread.kill()
			opticalThread.join()

		capture.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	lpr = LPR()
	lpr.videoCapturing()