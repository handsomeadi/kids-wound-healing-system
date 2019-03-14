import cv2
import requests
import numpy as np
import threading
import time

class IPcamCapture:
	def __init__(self, URL):
		self.Frame = []
		self.status = False
		self.isstop = False

		# Get stream of IPcam
		self.stream = requests.get(URL, auth=('admin', ''), stream=True)
		self.bytes = bytes()
	
	def start(self):
		# Use Thread to get image of IPcam
		print('IPcam Started!')
		threading.Thread(target=self.queryframe, args=()).start()

	def stop(self):
		# Stop the loop of queryframe
		self.isstop = True
		print('IPcam Stopped!')

	def getframe(self):
		# Get image frame (latest)
		return self.Frame

	def queryframe(self):
		bytes = ''
		for chunk in self.stream.iter_content(chunk_size=1024):
			
			if self.isstop:
				break
			
			self.bytes += chunk
			a = self.bytes.find(b'\xff\xd8')
			b = self.bytes.find(b'\xff\xd9')
			if a != -1 and b != -1:
				jpg = self.bytes[a:b+2]
				self.bytes = self.bytes[b+2:]
				self.Frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)	

    
if __name__ == '__main__':
	
	
	URL = 'http://192.168.0.195:3333/MJPEG.CGI'
	ipcam = IPcamCapture(URL)

	ipcam.start()

	time.sleep(1)


	while True:
		img = ipcam.getframe()

		cv2.imshow('Image', img)
		if cv2.waitKey(1) == 27:
			cv2.destroyAllWindows()
			ipcam.stop()
			break
