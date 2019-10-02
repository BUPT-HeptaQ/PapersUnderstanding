code ideas:

Step 1: Remove the background

Step 2: Grayscale

Step 3: Using cv2.canny to detect edges 

Step 4: Select the image area

Step 5: Use Hoffman to detect lines 

Step 6: Draw a deleted line

Step 7: Find x1, y1, x2, y2 for each column

Step 8: According to the gap interval, find the (x1, y1, x2, y2) where each train is located.

Step 9: Use the car1.h5 weight parameter obtained by Keras, use model.predict to perform prediction operation, 
predict the picture, and draw the picture.

Step 10: Forecast the video
