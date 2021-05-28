import time
import pickle
import numpy as np
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine

def name_time(person, names, prev_names):
    
    if person['names'] == []:
        person['names'] = names
        person['start_times'] = [time.time()] * len(names)
        person['elapsed_times'] = [0] * len(names)
    else:
        for name in names:
            if name != 'Unknown':
                if name in person['names']:
                    index = person['names'].index(name)
                    
                    if person['start_times'][index] != 0:
                        difference = time.time() - person['start_times'][index]

                        person['elapsed_times'][index] = person['elapsed_times'][index] + difference
                        person['start_times'][index] = time.time()
                    else:
                        person['start_times'][index] = time.time()

                else:
                    person['names'].append(name)
                    person['start_times'].append(time.time())
                    person['elapsed_times'].append(0)
        
        for name in prev_names:
            if name not in names and name in person['names']:
                index = person['names'].index(name)
                person['start_times'][index] = 0
    
    return person