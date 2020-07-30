import cv2
import numpy as np
import utlis
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'G:\\TesseractOCR\\tesseract.exe'


NOT_POSTED = True

#utlis.initializeTrackbars()
custom_config = r'--oem 3 --psm 3'

while True:

    image = cv2.imread('EpicGab.jpg')
    height, width, channels = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5,5), 1)
    #threshold = utlis.valTrackbars()
    threshold_image = cv2.Canny(blurred_image, 200, 120)
    kernel = np.ones((5,5))
    dialed_image = cv2.dilate(threshold_image, kernel, iterations = 3)
    threshold_image = cv2.erode(dialed_image, kernel, iterations = 1)

    imgContours = image.copy()
    imgBigContour = image.copy()
    image_warped = image.copy()
    image_warped_gray = image.copy()
    image_adapative = image.copy()
    image_warped_crop = image.copy()

    contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    biggest, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    
    if biggest.size != 0:
        biggest=utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[width, 0], [0, height],[width, height]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        image_warped = cv2.warpPerspective(image, matrix, (width, height))

        image_warped = image_warped[20:image_warped.shape[0] - 20, 20:image_warped.shape[1] - 20]
        image_warped = cv2.resize(image_warped,(width,height))
        #Crop Image (optional)
        #image_warped = image_warped[0:int(width/2), 0:height]
        image_warped_gray = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY)
        image_adapative = cv2.adaptiveThreshold(image_warped_gray, 255, 1, 1, 7, 2)
        image_adapative = cv2.bitwise_not(image_adapative)
        image_adapative = cv2.medianBlur(image_adapative, 3)
        image_warped_crop = image_warped_gray.copy()
        image_warped_crop = image_warped_crop[200:width-200, 0:int(height/2)]


    string = pytesseract.image_to_string(image_warped_crop, config = custom_config)
    data = []
    if(NOT_POSTED):
        for s in string.splitlines():
            s = str(s.split('\n'))
            data.append(s.lower())
        
        print("Full Name: " + str(data[0]) + " " + str(data[1]))
        print("UCID: " + str(data[4]))
        print("Predicted UCalgary Email: " + str(data[0]) + "." + str(data[1]) + "@ucalgary.ca")

        NOT_POSTED = False

    #cv2.imshow("grayscale", gray_image)
    #cv2.imshow("Blurred", blurred_image)
    #cv2.imshow("Threshold", threshold_image)
    #cv2.imshow("Image Contours", imgContours)
    cv2.imshow("Big Image Contours", imgBigContour)
    #cv2.imshow("Warped", image_warped)
    cv2.imshow("Warped Gray", image_warped_gray)
    cv2.imshow("Final", image_warped_crop)


    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()