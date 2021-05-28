import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from db import update_database
from mtcnn.mtcnn import MTCNN
from PIL import ImageFont, ImageDraw, Image
from keras_vggface.vggface import VGGFace
from text_functions import work_text
from face_functions import work_face, name_time
from vidgear.gears import CamGear

def show_output(img, result_text, temp_text, result_face, total_names):

    if result_text != []:
        for idx, res in enumerate(result_text):
            topleft = res[0][0]
            bottomright = res[0][2]
            cv2.rectangle(img, (int(topleft[0]), int(topleft[1])), (int(bottomright[0]), int(bottomright[1])), (0,255,0), 2)
            
            font = ImageFont.truetype('data/fonts/arial.ttf', 22)
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            w, h = font.getsize(temp_text[idx])

            draw.rectangle(( int(topleft[0]), int(topleft[1])-20, int(topleft[0]) + w, int(topleft[1]) + h -20), fill='black')
            draw.text((int(topleft[0]), int(topleft[1])-20),  temp_text[idx], font = font, fill = (0, 255, 255, 0))
            img = np.array(img_pil)

    if result_face != []:
        for idx, res in enumerate(result_face):
            x1, y1, width, height = res['box']
            x2, y2 = x1 + width, y1 + height
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.putText(img, total_names[idx], (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    cv2.imshow('Output', img)
    

if __name__ == '__main__':

    language = 'en'
    reader = easyocr.Reader([language])
    detector = MTCNN()
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    video = CamGear(source='https://www.youtube.com/watch?v=9Auq9mYxFEE&ab_channel=SkyNews', y_tube=True, logging=True).start()
    

    check = 0
    total_text = []
    person = {'names':[], 'elapsed_times':[], 'start_times':[]}
    prev_names = 'none'

    while True:

        frame = video.read()
        h,w,_ = frame.shape
        frame = cv2.resize(frame, dsize=(int(w*0.5), int(h*0.5)), interpolation=cv2.INTER_CUBIC)

        if frame is None:
            break
        
        '''
        Passing through processing functions
        '''
        if check % 10 == 0:
            
            ## Running deep learning models, getting text and face information
            result_text = reader.readtext(frame)
            result_face = detector.detect_faces(frame)
            
            ## Post Processing Text and Face information
            if result_text != []:
                total_text, temp_text = work_text(result_text, total_text, language)
                
            if result_face != []:
                names = work_face(result_face, frame, model)
                person = name_time(person, names, prev_names)
                prev_names = names
                
            ## Showing face and text detection/recognition
            if result_text != [] and result_face != []:
                show_output(frame, result_text, temp_text, result_face, names)
            
            
        '''
        Adding to database
        '''
        # if check % 30 == 0 and check != 0:
            # print(total_text, '\n', person['names'], '\n', person['elapsed_times'])
            
            # update_database(total_text, person, ids=0)
            
        
        '''
        Resetting arrays
        '''
            
        # if check % 500 == 0 and check != 0:
        #     ## Reset array and check
        #     total_text = []
        #     check = -1


        '''
        Ending current frame
        '''
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        
        check += 1
        
            

    cv2.destroyAllWindows()
    stream.stop()
