import os
import json
import time
import numpy as np
from db import update_database
from text_functions import work_text
from face_functions import name_time


def process_tensor_objects(frames):
  text_results = []
  names = []
  for f in frames:
    
    if 'Text' in f:
      data = f.split('|')
      text = data[-1]
      x1 = int(float(data[1]))
      y1 = int(float(data[2]))
      x2 = int(float(data[1])) + int(float(data[3]))
      y2 = int(float(data[2])) + int(float(data[4]))
      coords = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
      box = (coords, text)
      text_results.append(box)

    if 'Person' in f:
      name = f.split('|')[-1]
      name = name.replace('_', ' ')
      names.append(name)
      
  return text_results, names

if __name__ == '__main__':


  json_files = os.listdir('tensorRT_tempdata')
  jsons = []
  for files in json_files:
      with open('tensorRT_tempdata/' + files) as temp:
          temp = json.load(temp)
          jsons.append(temp)
  

  frame_check = 0 ## <---- update this to number after which databse will update
  total_text = []
  total_time = []
  total_names = []
  for frames in jsons:

    starting_time = time.time()

    ## Getting data from stream
    frames = frames['objects']
    
    ## Converting tensorRT data to current versions
    text_results, names = process_tensor_objects(frames)

    ## Measuring current time
    elapsed_time = time.time() - starting_time 
    elapsed_time = [elapsed_time] * len(names)
    
    ## Applying text functions to recognized text
    total_text, temp_text = work_text(text_results, total_text)
    total_names, total_time = name_time(names, total_names, elapsed_time, total_time)
    
    if frame_check % 3: ## <-- using three in this example since I have only 5 json files
      update_database(total_text, total_names, total_time)

    frame_check += 1
    

  