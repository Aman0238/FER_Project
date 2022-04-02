import cv2

status = "put text";
def textPrint(img, pred =0, plainText = False):
    if plainText == True:
        cv2.putText(img, f'pred: {int(pred)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        cv2.imshow('Result', img)
    elif plainText == False:
        if pred == 0:
            status = "Angry"
        elif pred == 1:
            status = "Disgust"
        elif pred == 2:
            status = "Fear"
        elif pred == 3:
            status = "Happy"
        elif pred == 4:
            status = "Neutral"
        elif pred == 5:
            status = "Sad"
        elif pred == 6:
            status = "Surprise"

        cv2.putText(img, status, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Result', img)
    else:
       print("nothing to show here!")








    cv2.waitKey()

