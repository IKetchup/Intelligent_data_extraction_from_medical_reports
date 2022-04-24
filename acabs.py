import cv2
import pytesseract
import numpy as np
import os


#tesseract dependencies
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'

cv2.resizeWindow('output', 400, 400)

# OCR in text block
def ACABS(imgg):
    img = cv2.imread(imgg)

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #crop 10 colums one the right and left of the image
    #avoid bugs with the segmentation algorithm on the border of the image
    h, w = gray.shape
    gray = np.array(gray[0:0 + h, 10: w - 10])

    # clean the image using otsu method with the inversed binarized image
    ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    boxs = create_bounding_box(th1, output)
    crop_top, crop_bottom = crop(boxs, gray, 0.10)
    text = ocr(gray, crop_top, crop_bottom, boxs)

    return text


# apply dilatation and erosion
# create bounding box
def create_bounding_box(thresh, output):
    boxs = []

    # assign a rectangle kernel size
    k1 = np.ones((15, 15), 'uint8')

    par_img = cv2.dilate(thresh, k1, iterations=3)
    par_img2 = cv2.erode(par_img, kernel = np.ones((5,5), 'uint8'), iterations=3)

    (contours, _) = cv2.findContours(par_img2.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    #create the bounding box
    for cnt in reversed(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        boxs.append([x, y, w, h])

        #draw bounding box on image
        #cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

    #cv2.imwrite("../segmentation/adaptative_reding_exemple/repport_9_box.jpg", output)
    return boxs

#define line of cropping
def crop(boxs, gray, percent):

    h, w = gray.shape

    #define percent of Y crop on the image
    top_crop_percent = percent
    bottom_crop_percent = percent
    crop_top = h * top_crop_percent
    crop_bottom = h * bottom_crop_percent

    #test
    output = gray.copy()

    #Adapt crop to avoid cropping in the middle of a bounding box
    #if the crop is made between Y and Y + H (bounding box) the crop become Y + H
    # box[0] : x
    # box[1] : y
    # box[2] : w
    # box[3] : h
    change_bottom = False
    for box in boxs:

        if (inbetween(box[1], crop_top, box[1] + box[3]) == crop_top):
            crop_top = box[1] + box[3]

        if (inbetween(box[1], h-crop_bottom, box[1] + box[3]) == h-crop_bottom):
            crop_bottom = box[1] + box[3]
            change_bottom = True

        cv2.rectangle(output, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 10)

    #round to avoid error
    crop_top = round(crop_top)
    crop_bottom = round(crop_bottom)

    #if bottom crop haven't change
    if not change_bottom:
        crop_bottom = h-crop_bottom

    #print bottom and top threshold lines on the image
    # bottom
    #output = cv2.line(output, (0, crop_bottom), (w, crop_bottom), (0, 0, 255), 5)
    # top
    #output = cv2.line(output, (0, crop_top), (w, crop_top), (0, 0, 255), 5)

    #cv2.imwrite("../segmentation/adaptative_reding_exemple/repport_9_box.jpg", output)

    return crop_top, crop_bottom


#find if val is between min and max
def inbetween(min,val,max):
    return sorted([min,val,max])[1]


#extract text for each bounding box
def ocr(gray, crop_top, crop_bottom, cropped_boxs):

        text = ""

        for box in cropped_boxs:
            #box[0] : x
            #box[1] : y
            #box[2] : w
            #box[3] : h

            if (box[1] > crop_top) and (box[1] + box[3] <= crop_bottom):
                ocr_box = np.array(gray[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])

                #check if array contain only 255
                #if(all(all(p == 255 for p in lines)for lines in ocr_box)):

                #check if 95% of the array is 255
                flattened = np.ravel(ocr_box)
                if not (np.sum((flattened == 255)) / len(flattened) > 0.95):

                    #cv2.imshow('ok', ocr_box)
                    #cv2.waitKey(0)
                    data = (pytesseract.image_to_string(ocr_box, lang='fra'))

                    #\jump used as a delimiter of each text block
                    #Used in the NER annotator software

                    data = data.replace(' \n\n', '')
                    data = data.replace(' \n\x0c', '')
                    data = data.replace('\x0c', '')

                    text += data

        return text
