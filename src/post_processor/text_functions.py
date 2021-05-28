import re
import cv2
import enchant
import numpy as np
import arabic_reshaper
from difflib import SequenceMatcher
from spellchecker import SpellChecker
from bidi.algorithm import get_display

def track(total_text, new_text, language):

    total_text = total_text.split(' ')
    new_text = new_text.split(' ')
    both = list(set(new_text).intersection(total_text))
    indices_A = [total_text.index(x) for x in both]
    indices_A.sort()

    try:
        
        check = indices_A[int(len(indices_A)/2)]
        word = total_text[check]
        check = [i for i, j in enumerate(total_text) if j == word]
        check = check[-1]

        if language == 'en':
            output1 = ' '.join(total_text[:check])
        else:
            output1 = ' '.join(total_text[check:])
        
        check = [i for i, j in enumerate(new_text) if j == word]
        check = check[-1]

        if language == 'en':
            output2 = ' '.join(new_text[check:])
        else:
            output2 = ' '.join(new_text[:check])
        
        if language == 'en':
            output = output1 + ' ' + output2
        else:
            output = output2 + ' ' + output1

    except:

        total_text = new_text
        output = ' '.join(total_text)
    
    return output

def return_text(result, language):

    text = []

    for idx, res in enumerate(result):
        
        if language == 'en':
            res = re.sub('[^A-Za-z0-9]+', ' ', res[1])
            res = res.lower()
            
        else:
            res = res[1]
            res = arabic_reshaper.reshape(res)
            res = get_display(res)
            
        text.append(res)

    return result, text

def compare_rect(text, result):
    
    coords = []
    
    for idx, res in enumerate(result):
        
        coords.append([int(res[0][0][0]), int(res[0][0][1]), int(res[0][2][0]), int(res[0][2][1]), text[idx]])
    
    coords = sorted(coords, key=lambda x: x[1])
    
    prev_y1 = coords[0][1]
    temp = []
    temp.append(coords[0])
    coords2 = []
    for coord in coords[1:]:
        new_y1 = coord[1]
        dist = abs(new_y1 - prev_y1)
        
        if dist <= 5:
            temp.append(coord)
            prev_y1 = coord[1]
        else:
            coords2.append(temp)
            temp = []
            temp.append(coord)
            prev_y1 = coord[1]
    coords2.append(temp)        
        
    
    new_coords = []
    for coords in coords2:
        coords = sorted(coords, key=lambda x: x[0])
        new_coords.append(coords)
    coords2 = new_coords
    
    # new_coords = []
    # temp = []
    # for coords in coords2:
        
    #     if len(coords) == 1:    
    #         new_coords.append(coords)
    #         continue
        
    #     prev_x1 = coords[0][2]
    #     temp.append(coords[0])

    #     for coord in coords[1:]:
    #         new_x1 = coord[0]
    #         diff = abs(prev_x1 - new_x1)

    #         if diff <= 50:
    #             temp.append(coord)
    #             prev_x1 = coord[2]
    #         else:
    #             new_coords.append(temp)
    #             temp = []
    #             temp.append(coord)
    #             prev_x1 = coord[2]
    #     new_coords.append(temp)
    #     temp = []
    # coords2 = new_coords

    text_array = []
    final_coords = []

    for coords in coords2:
        string = ''
        
        for words in coords:
            words = words[-1]
            string = string + words + ' '
        string = string[:-1]
        text_array.append(string)
        final_coords.append([[coords[0][0], coords[0][1]], [coords[-1][2], coords[-1][3]], string])

    return text_array, final_coords

def compare_text(text, total_text, language):

    
    if total_text == []:
        return text

    new_text = []
    for strings in total_text:
        
        index = -1
        for idx, strings2 in enumerate(text):
            
            both = len(list(set(strings2.split(' ')).intersection(strings.split(' '))))
            check = len(strings2.split(' '))

            if both < check:
                toast = check
                check = both
                both = toast

            try:
                percentage = (check / both) * 100
            except:
                percentage = 0

            if percentage >= 60:
                index = idx

        if index != -1:
            
            temp_text = track(strings, text[index], language)
            
            try:
                if temp_text[0] == ' ':
                    temp_text = temp_text[1:]
                elif temp_text[-1] == ' ':
                    temp_text = temp_text[:-1]
            except:
                continue

            new_text.append(temp_text)
            del text[index]
            
        else:
            try:
                if strings[0] == ' ':
                    strings = strings[1:]
                elif strings[-1] == ' ':
                    strings = strings[:-1]
            except:
                continue

            new_text.append(strings)

    for strings in text:
        check = 1
        for strings2 in total_text:
            ratio = SequenceMatcher(None, strings, strings2).ratio()
            if ratio >= 0.5:
                check = 0
                break
        
        if check == 1:
            new_text.append(strings)
        

    return new_text

def work_text(result, total_text, language='en'):

    if language != 'en' and language != 'ur':
        print('Please enter correct language')
        return None

    result, temp_text = return_text(result, language)
    if len(temp_text) > 1:
        temp_text, _ = compare_rect(temp_text, result)

    total_text = compare_text(temp_text, total_text, language)
    total_text = list(set(total_text))
    
    return total_text, temp_text