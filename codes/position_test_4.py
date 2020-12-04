#!/usr/bin/python3
"""
Vishnu... Thank you for electronics.

Author :- Manas Kumar Mishra.
Co-author: Maayank Navneet Mehta
"""
"""
Task :- To utilize position_test_3.py for VIMAN.
"""
"""
Modifications made wrt v3:
	- Show video with keypoints
	- System does not stall on not finding any object
	- Summation of displacements between frames 

TODO:
	0: Show video with keypoints
	1: System does not stall on not finding any object
	2: Ouput of the system is the displacement wrt initial position
		2.1: Summation of displacements between frames 
"""


import numpy as np
import cv2
import time

# transformation of body wrt global
Tg = np.eye(4)

def HtransformationMat(R,t):
	c = np.c_[R,t]
	
	# prespective vector and scaling value.
	scale =1
	eta = np.array([0,0,0,scale])
	
	htrans = np.vstack([c,eta])
	
	return htrans

def main():
	global Tg
	# camera parameters (camera matrics)
	# Change this matrix according to your camera.
	K = np.array([1189.46, 0.0, 805.49, 0.0, 1191.78, 597.44, 0.0, 0.0, 1.0]).reshape(3,3)

	# Capturing the video through the webcam
	cap = cv2.VideoCapture(0)

	# ORB OBJECT
	orb = cv2.ORB_create()

	#Capture inital frame.
	ret0, oldframe = cap.read()
	oldframe = cv2.medianBlur(oldframe, 5)

	oldgray = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)

	# keypoints and descriptors
	kp0, des0 = orb.detectAndCompute(oldgray, None)

	while True:
		# read the video frame by frame
		ret, newframe = cap.read()
		newframe = cv2.medianBlur(newframe, 5)
		
		newgray = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
		
		# keypoints and descriptors
		kp1, des1 = orb.detectAndCompute(newgray, None)
		
		# Features
		# Matcher
		FLANN_INDEX_LSH = 6
		index_params= dict(algorithm = FLANN_INDEX_LSH,
							table_number = 12, # 6
							key_size = 20,     # 10
							multi_probe_level = 2) #1 also possible

		search_params = dict(checks=50)   # or pass empty dictionary
		flann = cv2.FlannBasedMatcher(index_params,search_params)
		if des1 is not None and des0 is not None:
			if des0.size >= 2 and des1.size >= 2:
				try:
					matches = flann.knnMatch(des0,des1,k=2)
					good = []
					pt0 = []
					pt1 = []
					
					for i,pair in enumerate(matches):
						try:
							m,n = pair
							if m.distance < 0.7*n.distance:
								good.append(m)
								pt1.append(kp1[m.trainIdx].pt)
								pt0.append(kp0[m.queryIdx].pt)
						except ValueError:
							pass
					
					pt0 = np.int32(pt0)
					pt1 = np.int32(pt1)
					
					if(np.shape(pt0)[0]!=0 and np.shape(pt1)[0]!=0):
						F, mask1 = cv2.findFundamentalMat(pt0, 
														pt1, 
														method=cv2.FM_RANSAC ) #method= cv2.FM_LWEDS also a possiblity.
						if F is not None:
							try:
								E = K.T.dot(F).dot(K)
								p, R, t, mask = cv2.recoverPose(E, pt0, pt1, focal=1, pp=(0., 0.))
								tt = HtransformationMat(R,t)
							except ValueError:
								tt = np.eye(4)
								pass
				except:
					print("Did not find keypoints")
					tt = np.eye(4)
		
		# summation
		Tg = Tg @ tt
		# Tg = tt @ Tg

		# display results
		print("x: ", Tg[0][3], " | y: ", Tg[1][3], " | z: ", Tg[2][3])
		print('\n')

		showframe = cv2.drawKeypoints(newgray, kp1, None, color=(0, 255, 0))
		cv2.imshow('VIMAN_VO',showframe)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()

# Thank you.