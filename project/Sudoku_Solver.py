import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils
from solver import *
from tkinter import *
from PIL import ImageTk
import os

root = Tk()
root.title("Welcome to Sudoku Solver")
root.geometry('1000x1000')

bg = ImageTk.PhotoImage(file = 'bg-img.jpg')
canvas1 = Canvas(root, width = 1000, height = 1000)
canvas1.pack(fill = "both", expand = True)
canvas1.create_image( 0, 0, image = bg, anchor = "nw")

lbl1 = Label(root, text = "\n\nClick the button to capture sudoku \n S - to capture \n Q - to exit.\n\n")
lbl1.place(x=400, y=50)

def imgCapture():
    lbl1.configure(text = "\nThe button was clicked to capture sudoku using camera.\n")
    btn1['text'] = 'Was Clicked'
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            print(check)
            print(frame)
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'): 
                cv2.imwrite(filename='sudoku.jpg', img=frame)
                webcam.release()
                img_new = cv2.imread('sudoku.jpg')
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
            
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
            
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

btn1 = Button(root, text = "Click me to capture" ,
             fg = "blue", command=imgCapture)
btn1.place(x=440, y=200)

lbl2 = Label(root, text = "\nClick the button to get output of the sudoku solver.\n")
lbl2.place(x=360, y=350)

def processSudoku():
    lbl2.configure(text = "\nThe button was clicked to get the solved sudoku.\n")
    btn2['text'] = 'Was Clicked'

    classes = np.arange(0, 10)

    model = load_model('model-OCR.h5')

    input_size = 48


    def get_perspective(img, location, height = 900, width = 900):
        pts1 = np.float32([location[0], location[3], location[1], location[2]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, matrix, (width, height))
        return result

    def get_InvPerspective(img, masked_num, location, height = 900, width = 900):

        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = np.float32([location[0], location[3], location[1], location[2]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
        return result

    def find_board(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
        edged = cv2.Canny(bfilter, 30, 180)
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours  = imutils.grab_contours(keypoints)

        newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        location = None

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 15, True)
            if len(approx) == 4:
                location = approx
                break
        result = get_perspective(img, location)
        return result, location

    def split_boxes(board):

        rows = np.vsplit(board,9)
        boxes = []
        for r in rows:
            cols = np.hsplit(r,9)
            for box in cols:
                box = cv2.resize(box, (input_size, input_size))/255.0
                boxes.append(box)
        cv2.destroyAllWindows()
        return boxes

    def displayNumbers(img, numbers, color=(0, 255, 0)):
        
        W = int(img.shape[1]/9)
        H = int(img.shape[0]/9)
        for i in range (9):
            for j in range (9):
                if numbers[(j*9)+i] !=0:
                    cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int((j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
        return img

    img = cv2.imread('sudoku.jpg')

    board, location = find_board(img)


    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    rois = split_boxes(gray)
    rois = np.array(rois).reshape(-1, input_size, input_size, 1)

    prediction = model.predict(rois)

    predicted_numbers = []
    for i in prediction: 
        index = (np.argmax(i))
        predicted_number = classes[index]
        predicted_numbers.append(predicted_number)

    board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)

    try:
        solved_board_nums = get_board(board_num)
        binArr = np.where(np.array(predicted_numbers)>0, 0, 1)
        flat_solved_board_nums = solved_board_nums.flatten()*binArr
        mask = np.zeros_like(board)
        solved_board_mask = displayNumbers(mask, flat_solved_board_nums)
        inv = get_InvPerspective(img, solved_board_mask, location)
        combined = cv2.addWeighted(img, 0.7, inv, 1, 0)
        cv2.imshow("Final result", combined)
        

    except:
        print("Solution doesn't exist. Model misread digits.")

    cv2.imshow("Input image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

btn2 = Button(root, text = "Click me to get solved sudoku" ,
             fg = "blue", command=processSudoku)
btn2.place(x=420, y=450)

def imgDelete():
    btn3['text'] = 'Image Deleted'
    os.remove('sudoku.jpg')


btn3 = Button(root, text = "Click me to delete the captured image" ,
             fg = "red", command=imgDelete)
btn3.place(x=405, y=570)

root.mainloop()