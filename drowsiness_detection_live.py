# -*- coding: utf-8 -*-


# import the necessary packages
from scipy.spatial import distance as dist # to compute euclidiean distance
from imutils.video import VideoStream
from imutils import face_utils
from util.activity_tracker import ActivityTracker
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
#------------- For Twilio ----------------
import os
from twilio.rest import Client
path="alarm.wav"

# Twilio Account SID and Auth Token
account_sid = "ACd67e3fb1f0864df4c67effc0a4233ffb"	#TWILIO ACCOUNT SID
auth_token = "a0bb8d940010f77a97a6e01fea7115f7"	#TWILIO_AUTH_TOKEN






client = Client(account_sid, auth_token)
# defining eye aspect ratio according to research paper
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[3], mouth[9]) # 51, 59
	# B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = A  /  C

	# return the mouth aspect ratio
	return mar





# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0,help="index of webcam on system") # integer controls the index of your built-in webcam/USB camera
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm

# Declaring value of threshold for eye aspect ratio
EYE_AR_THRESH = 0.25
# Declaring value of threshold for number of consecutive frames
EYE_AR_CONSEC_FRAMES = 25 

MOUTH_AR_THRESH = 0.85

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

(mStart, mEnd) = (49, 68)

# start the video stream thread
print("starting video stream thread...")
vs = VideoStream(src=1).start()
time.sleep(1.0)
# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=1080)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
    # loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		mouth = shape[mStart:mEnd]
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		mouthMAR = mouth_aspect_ratio(mouth)

		mar = mouthMAR
		# compute the convex hull for the mouth, then
		# visualize the mouth
		mouthHull = cv2.convexHull(mouth)
		
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);cv2.putText(frame, "****************ALERT!****************", (10,325),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
		# otherwise, the eye aspect ratio is not below the blink
				playsound.playsound(path)
				
		# threshold, so reset the counter and alarm
		
		elif mar > MOUTH_AR_THRESH:
			cv2.putText(frame, "Mouth is Open!", (30,60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
			COUNTER += 1
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# draw an alarm on the frame
				
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);cv2.putText(frame, "****************ALERT!****************", (10,325),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
		# otherwise, the eye aspect ratio is not below the blink
				playsound.playsound(path)
		
		else:
			COUNTER = 0
			ALARM_ON = False
        # draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	size = frame.shape

	tracker = ActivityTracker()

	image_points = np.array([
									(shape[33, :]),     # Nose tip
									(shape[8,  :]),     # Chin
									(shape[36, :]),     # Left eye left corner
									(shape[45, :]),     # Right eye right corne
									(shape[48, :]),     # Left Mouth corner
									(shape[54, :])      # Right mouth corner
								], dtype="double")


	many_points = np.array([(shape[i,:]) for i in range(68)], dtype="double")
								# 	(shape[0, :]),     
								# 	(shape[3, :]),     
								# 	(shape[13, :]),    
								# 	(shape[16, :]),    
								# 	(shape[29, :]),    
								# 	(shape[30, :]),    
								# 	(shape[17, :]),     
								# 	(shape[21, :]),     
								# 	(shape[26, :]),     
								# 	(shape[22, :]),     
								# 	(shape[33, :]),     # Nose tip
								# 	(shape[8,  :]),     # Chin
								# 	(shape[36, :]),     # Left eye left corner
								# 	(shape[45, :]),     # Right eye right corne
								# 	(shape[48, :]),     # Left Mouth corner
								# 	(shape[54, :])      # Right mouth corner
								# ], dtype="double")
		# 3D model points.
	model_points = np.array([
									(0.0, 0.0, 0.0),             # Nose tip
									(0.0, -330.0, -65.0),        # Chin
									(-225.0, 170.0, -135.0),     # Left eye left corner
									(225.0, 170.0, -135.0),      # Right eye right corne
									(-150.0, -150.0, -125.0),    # Left Mouth corner
									(150.0, -150.0, -125.0)      # Right mouth corner                     
								])
		
	
	focal_length = size[1]
	center = (size[1]/2, size[0]/2)
	camera_matrix = np.array(
								[[focal_length, 0, center[0]],
								[0, focal_length, center[1]],
								[0, 0, 1]], dtype = "double"
								)


	dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
	(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

	(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)


	for p in many_points:
		cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0,255,0), -1)

	p1 = ( int(image_points[0][0]), int(image_points[0][1]))
	p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
		
	diff = (p1[0] - p2[0], p1[1] - p2[1])
	sq_dist = diff[0] * diff[0] + diff[1] * diff[1]
	
	if sq_dist > 30000:
		tracker.start_activity("Distracted")
		cv2.putText(frame, "DISTRACTION ALERT!", (30, 90),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	else:
		tracker.start_activity("Focused")
		cv2.arrowedLine(frame, p1, p2, (0,255,0), 2)
		
			



	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		
		break
	if key == ord("w"):
		body="You have been selected as an emergency number by user. User is currently feeling drowsiness. Please contact him asap so that he can travel safely."

		client.api.account.messages.create(to="+917972795557",from_="+13156878133",body=body)
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

"""
To run from Terminal:
    python drowsiness_detection_live.py --shape-predictor
    
"""













