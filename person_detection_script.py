#!/usr/bin/python3
#-*-coding: utf-8-*-

# import the necessary packages
from os import path
import sys
import cv2
import can
import numpy as np

def main(prototxt_filepath, caffenet_filepath):
	# CAN-bus initialization block
	can_bus = can.Bus(interface='socketcan', channel='vcan0',
						 receive_own_messages=True)
	can_msg_lost_camera = can.Message(arbitration_id=0x018, is_extended_id=True, 
					  data=[0, 0, 0, 0, 0, 0, 0, 0])
	can_msg_person_is_here = can.Message(arbitration_id=0x018, is_extended_id=True,
					     data=[1, 0, 0, 0, 0, 0, 0, 0])
	can_msg_person_is_nearly = can.Message(arbitration_id=0x018, is_extended_id=True,
					       data=[1, 1, 0, 0, 0, 0, 0, 0])

	# Camera initialization block
	print("Connecting to camera...")
	cap = cv2.VideoCapture(0)
	if cap.isOpened():
		print("Successfull")
	else:
		print("Error, camera is not available")
		try:
			can_bus.send(can_msg_lost_camera)
		except:
			print("CAN bus is not available")
		return
	
	# Neural Network initialization block
	net_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
					"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
					"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
					"sofa", "train", "tvmonitor"]
	colors_for_net = np.random.uniform(0, 255, size=(len(net_classes), 3))
	limit_for_confidence = 0.8
	neural_net = cv2.dnn.readNetFromCaffe(prototxt_filepath, caffenet_filepath)
	cnt_inc = 0
	# Main cycle block
	while cap.isOpened():
		# loop the frame and convert it to RGB-format
		_, frame = cap.read()
		flag_inc = False

		# Grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
					     0.007843, (300, 300), 127.5)
		
		# Pass the blob through the network and obtain the detections and predictions
		neural_net.setInput(blob)
		detections = neural_net.forward()
		start_x = start_y = end_x = end_y = None

		# Check all detections
		for i in np.arange(0, detections.shape[2]):
			# Extract the confidence (i.e., probability) associated with the predictions
			confidence = detections[0, 0, i, 2]
			# Filter out weak detections by ensuring the confidence is greater than 
			# the minimum confidence
			if confidence > limit_for_confidence:
				# Extract the index of the class labels from detections, then compute
				# the (x, y)-coordinates of the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(start_x, start_y, end_x, end_y) = box.astype("int")
				# Draw the bounding boxes for everything person
				if net_classes[idx] == 'person':
					if cnt_inc < 5:
						cnt_inc += 1
					label = "{}: {:.2f}%".format(net_classes[idx], confidence*100)
					if(end_y - start_y) > 0.5*h or (end_x - start_x) > 0.5*w:
						sending_msg = can_msg_person_is_nearly
					else:
						sending_msg = can_msg_person_is_here
					cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), colors_for_net[idx], 2)
					y = start_y - 15 if start_y - 15 > 15 else start_y + 15
					cv2.putText(frame, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
									colors_for_net[idx], 2)
		if cnt_inc > 4:
			try:
				can_bus.send(sending_msg)
				print("Message sent on {}".format(can_bus.channel_info))
			except:
				print("Message NOT sent")
			cnt_inc = 0
		cv2.imshow("abc", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	print("Close connection")
	cap.release()

if __name__ == "__main__":
	script_dir = path.dirname(path.realpath(__file__))
	caffe_net_filepath = path.join(script_dir, 'MobileNetSSD_deploy.caffemodel')
	caffe_net_filepath = path.abspath(caffe_net_filepath)
	proto_filepath = path.join(script_dir, 'MobileNetSSD_deploy.prototxt.txt')
	proto_filepath = path.abspath(proto_filepath)
	main(caffenet_filepath=caffe_net_filepath, prototxt_filepath=proto_filepath)
