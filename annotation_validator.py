# Annotations validation tool 
"""
Next frame
previous frame
clear annnotations
draw annotations
save annotations

"""

import cv2
import glob
from numpy.testing._private.utils import import_nose
import pandas as pd
from pandas import DataFrame
import numpy as np
from natsort import natsorted
from pathlib import Path
import os
from tkinter import *
from tkinter import filedialog



def read_annot_file():
    """
    Read Annotation file
    """
    annot_file_path = filedialog.askopenfilename(initialdir = "/Downloads",
										title = "Select annotation file",
										filetypes = (('text files', '*.txt'),
														('All files', '*.*')))
    return annot_file_path

def read_img_files():
    """
    Read sequence of images
    """
    img_paths = filedialog.askopenfilenames(initialdir = "/Downloads", title = "Select Trajectory Images File", filetypes = (('Image files', '*.jpg'), ('All files', '*.*')))
    return img_paths


def annot_file_to_df(annotation_file):
    """
    Annotation file read ad return dataframe
    """
    df = pd.read_csv(annotation_file, delimiter=',', header=None)
    df.columns=['FRAME','ID','BB_LEFT','BB_TOP','BB_WIDTH','BB_HEIGHT','CONF','X','Y','Z']
    ball_frames = df.loc[df['ID'] == 1]
    return ball_frames


def extract_img_annot(df, frame_num):
    """
    Extract annotations point form txt files and send send to draw
    """
    # flag = []
    # flag.append(frame_num)
    # return df.loc[df['FRAME'].isin(flag)]
    row = df.loc[df['FRAME'] == frame_num] 
    x1 = row['BB_LEFT']
    y1 = row['BB_TOP']
    x2 = row['BB_WIDTH']
    y2 = row['BB_HEIGHT']
    return x1, y1, x2, y2, row.shape[0]
    
def extract_df_img_annot(df, frame_num):
    """
    Extract multiple rows based on frame_num, return a df and send send to draw
    """
    flag = []
    flag.append(frame_num)
    return df.loc[df['FRAME'].isin(flag)]

def draw_annot(x1, y1, x2, y2, image):
    """
    Draw bbox on images
    """
    color = (255, 0, 0)
    thickness = 1
    start_point = (x1, y1)
    end_point = (x2, y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def update_df(df, img_name, boxes):
    """
    add/update dataframe with annotation
    'FRAME','ID','BB_LEFT','BB_TOP','BB_WIDTH','BB_HEIGHT','CONF','X','Y','Z'
    """
    new_row = {'FRAME':img_name, 'ID':1, 'BB_LEFT':boxes[-2][0], 'BB_TOP':boxes[-2][1], 'BB_WIDTH':boxes[-1][0] , 'BB_HEIGHT':boxes[-1][1] , 'CONF':1, 'X':-1, 'Y':-1, 'Z':-1}
    #append row to the dataframe
    df = df.append(new_row, ignore_index=True)
    return df

def save_annot(df, annot_path):
    """
    Save annotations to txt file
    """
    df.to_csv(annot_path, header=None, index=None, sep=',', mode='w')
    
def check_annot(df, img_name):
    """
    return bool whether annotation available or not in df  
    """
    flag = []
    flag.append(img_name)
    #print(df.loc[df.FRAME == img_name]["FRAME"].values)
    #print(df.loc[df['FRAME'].isin(flag)])
    #print(df.loc[df['FRAME'].isin(flag)].shape[0])
    #print(df.loc[df['FRAME'].isin(flag)])
    return df.loc[df['FRAME'].isin(flag)].shape[0] != 0
    #df.loc[df.FRAME == img_name]["FRAME"].values == img_name

def clear_annot(df, img_name):
    """
    Remove / clear annotation from an image
    """
    return df[df.FRAME != img_name]

def on_mouse(event, x, y, flags, params):
    """
    Draw bbox on image.
    Access x1, y1, x2, y2
    """
    # global img

    if event == cv2.EVENT_LBUTTONDOWN:
        #print('Start Mouse Position: '+str(x)+', '+str(y))
        sbox = [x, y]
        boxes.append(sbox)

    elif event == cv2.EVENT_LBUTTONUP:
        #print('End Mouse Position: '+str(x)+', '+str(y))
        ebox = [x, y]
        boxes.append(ebox)

#---------------------------------------------------------------------------------
toplevel = Tk()
toplevel.withdraw()


while True:
    #Read annotation and image sequence
    annot = read_annot_file()
    img_paths = natsorted(list(read_img_files()))
    print(annot)

    # annotation dataframe
    annot_df = annot_file_to_df(annot)

    # create view
    cv2.namedWindow(annot, cv2.WINDOW_NORMAL)
    cv2.moveWindow(annot, 250, 150)
    cv2.namedWindow('controls')
    cv2.moveWindow('controls',250,50)

    # Controls display window 
    controls = np.zeros((50,1150),np.uint8)
    cv2.putText(controls, "W/w: Play, S/s: Stay/Pause, A/a: Prev, D/d: Next, U/u: Update, R/r: Save, C/c: Clear annotations, Esc: Exit", (40,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

    # Flags
    status = 'stay'
    tot_imgs = len(img_paths)
    iter_index = 0
    boxes = []

    while True:
        if (iter_index == tot_imgs) or (iter_index < 0):
            iter_index = 0

        # access Image name
        img_name = int(Path(img_paths[iter_index]).stem)

        # read image
        frame = cv2.imread(img_paths[iter_index])

        # condition whether annotation available or not
        annot_flag = check_annot(annot_df, img_name)
        #print(annot_flag)
        
        try:
            # condition whether annotation available or not
            if annot_flag:
                #print("plain if")
                x1, y1, x2, y2, tot_anot = extract_img_annot(annot_df, img_name)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                cv2.putText(frame, f"Image: {str(img_name)}, Annotations: {tot_anot}", (40,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                cv2.imshow(annot, frame)
            else:
                #print("first else")
                cv2.putText(frame, f"Image: {str(img_name)}", (40,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                cv2.imshow(annot,frame)
        except Exception as e:
            # print(e)
            df = extract_df_img_annot(annot_df, img_name)
            try:
                #print(df)
                #print(list(df.index))
                for index, row in df.iterrows():
                    x1 = row['BB_LEFT']
                    y1 = row['BB_TOP']
                    x2 = row['BB_WIDTH']
                    y2 = row['BB_HEIGHT']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                    cv2.putText(frame, f"Image: {str(img_name)}, Annotations: {df.shape[0]}, index: {index}", (40,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                    cv2.imshow(annot, frame)
                    key = { ord('c'):'clear_annotation', ord('C'):'clear_annotation'}[cv2.waitKey(0)]
                    if key =='clear_annotation':
                        keep = list(df.index)
                        keep.remove(index)
                        #annot_df = clear_annot(annot_df, img_name)
                        # Drop Colums a & b from dfObj in place
                        annot_df.drop(keep, inplace=True)
                        df.drop(keep, inplace=True)
                        #print(df)
                        print(f"Annotations cleared on image: {img_name}")
                    else:
                        pass
                    #print (row["FRAME"], row["BB_LEFT"], row["BB_TOP"])
                #pass
                #for i in range(len(df)) :
                #    print(df.loc[i, "BB_LEFT"], df.loc[i, "BB_TOP"], df.loc[i, "BB_WIDTH"], df.loc[i, "BB_HEIGHT"])
                
                #annot_df = clear_annot(annot_df, img_name)
            except:
                pass
            cv2.putText(frame, f"Image: {str(img_name)}, Multi cleaned", (40,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
            cv2.imshow(annot,frame)
            #print("first try, except")

        cv2.imshow("controls",controls)
        cv2.setMouseCallback(annot, on_mouse)
        try:
            status = { ord('s'):'stay', ord('S'):'stay',
            ord('w'):'play', ord('W'):'play',
            ord('a'):'prev_frame', ord('A'):'prev_frame',
            ord('d'):'next_frame', ord('D'):'next_frame',
            ord('u'):'update annotations', ord('U'):'update annotations',
            ord('r'):'save annotation', ord('R'):'save annotation',
            ord('c'):'clear_annotation', ord('C'):'clear_annotation',
            -1: status, 
            27: 'exit'}[cv2.waitKey(0)]
            
            # keys
            if status == 'play':
                continue
            if status == 'stay':
                iter_index = iter_index
            if status == 'exit':
                break
            if status =='prev_frame':
                iter_index -= 1
                status = 'stay'
            if status =='next_frame':
                iter_index += 1
                status = 'stay'
            if status == 'update annotations':
                annot_df = update_df(annot_df, img_name, boxes)
                status = 'stay'
            if status =='save annotation':
                save_annot(annot_df, annot)
                annot_df = annot_file_to_df(annot)
                status = 'stay'
            if status =='clear_annotation':
                annot_df = clear_annot(annot_df, img_name)
                print(f"Annotations cleared on image: {img_name}")
                status ='stay'
            if status == 'exit':
                break
        except KeyError:
            print("Invalid Key was pressed")
    cv2.destroyWindow(annot)


# Mouse click event
# https://stackoverflow.com/questions/22140880/drawing-rectangle-or-line-using-mouse-events-in-open-cv-using-python