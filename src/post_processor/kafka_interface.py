import os
import json
import time
import numpy as np
from text_functions import work_text
from face_functions import name_time
from time import sleep
from json import dumps, loads
from kafka import KafkaConsumer

def process_tensor_objects(detections):
  detected_text = []
  detected_persons = []
  for detection in detections:
    data = detection.split('|')
    if 'Text' in data:
      text = data[7]
      x1 = int(float(data[1]))
      y1 = int(float(data[2]))
      x2 = int(float(data[1])) + int(float(data[3]))
      y2 = int(float(data[2])) + int(float(data[4]))
      coords = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
      detected_text.append((coords, text))

    if 'Person' in data:
      name = data[7]
      name = name.replace('_', ' ')
      detected_persons.append(name)
      
  return detected_text, detected_persons

# kafka streaming port 
KAFKA_STREAM_PORT = 8765

# to be taken from database for current view
channels_to_handle = [
    'FOX_NEWS_1',
    'FOX_NEWS_2',
    'CNN_NEWS_1',
    'CNN_NEWS_2'
]

topic = 'media_app_analytics'
detection_consumer = KafkaConsumer(
    topic,
    bootstrap_servers=['10.12.42.157:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='media_app_analytics',
    value_deserializer=lambda x: loads(x.decode('utf-8')))

# remove all previous msgs
detection_consumer.poll()
detection_consumer.seek_to_end()

total_text = {}
total_names = {}
for channel_id in channels_to_handle:
    total_text[channel_id] = []
    total_names[channel_id] = {}

for detection in detection_consumer:
    channel_id = detection.value['sensorId']
    if detection.value['sensorId'] in channels_to_handle:
        detections = detection.value['objects']

        # convert analytics data received from deepstream to understandable format
        text, persons = process_tensor_objects(detections)
        
        # analyse and fix text
        total_text[channel_id], temp_text = work_text(text, total_text[channel_id])
        # total_names[channel_id] = name_time(names, total_names[channel_id])
        
        # # if frame_check % 3: ## <-- using three in this example since I have only 5 json files
        # #   update_database(total_text, total_names, total_time)
        if detection.value['sensorId'] == 'CNN_NEWS_1':
            print ('total_text', total_text['CNN_NEWS_1'])
            # print('total_names', total_names['CNN_NEWS_1'])

        # frame_check += 1