import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import numpy as np
import os
import math
import time
import random

# Enable GPU Memory Growth
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_border(mask_land, mask_sky):
    """Get horizon border image from land and sky mask"""
    # Convert Colorspace to Grayscale
    mask_land = mask_land[:,:]
    mask_sky = mask_sky[:,:]
    # Get Horizon Border Using Dilation and Bitwise AND
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    land_dilated = cv2.dilate(mask_land, kernel)
    sky_dilated = cv2.dilate(mask_sky, kernel)
    border = cv2.bitwise_and(land_dilated, sky_dilated)

    return border

def get_horizon_line(border, ransac=False):
    """Get horizon line equation from border image"""
    # Get border data in x,y format
    y = np.argmax(border, axis=0)
    x = np.arange(len(y))
    border_data = np.vstack([x, y]).T

    # Remove 0 from border data
    border_data = border_data[border_data[:, -1] != 0]
    
    if ransac :
        n = border_data.shape[0]
        m = 0
        c = 0
        
        bestScore = -1
        for k in range(50):
            i1,i2 = random.sample(range(n), 2)
            p1 = border_data[i1]
            p2 = border_data[i2]

            dp = p1-p2
            dp = dp * (1./np.linalg.norm(dp))

            score = 0
            for i in range(n):
                v = border_data[i] - p1
                dis = v[1]*dp[0]-v[0]*dp[1]
        
                if math.fabs(dis)<0.7:
                    score += 1
            
            if score > bestScore:
                m = float(dp[1]) / float(dp[0])
                c = -m * p1[0] + p1[1]
                bestScore = score
    else:
        # Linear Regression using border data
        # y = m*x+c
        x = border_data[:,0]
        y = border_data[:,1]

        X = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(X, y, rcond=None)[0]
    
    return m, c

def get_roll_pitch(m, c, image_height, image_width):
    """Get roll and pitch from horizon line equation"""
    # Convert slope (m) to roll degrees
    roll = math.degrees(math.atan(m))

    # Get pitch
    pitch = ((m*(image_width/2)+c)-(image_width/2))/(image_width/2)*100
    
    return roll, pitch

def draw_horizon_line(img, m, c, scale, color=(0,0,0)):
    """Draw horizon line on image"""
    image_height = img.shape[0]
    image_width = img.shape[1]

    c = scale*c

    pt1 = (0, int(m*0+c))
    pt2 = (image_width, int(m*image_width+c))

    cv2.line(img, pt1, pt2, color, 2)

    return img

# Metric Function
class MaxMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

# Loss Function
def dice_loss(y_true, y_pred, num_classes=2):
    smooth=tf.keras.backend.epsilon()
    dice=0
    for index in range(num_classes):
        y_true_f = tf.keras.backend.flatten(y_true[:,:,:,index])
        y_pred_f = tf.keras.backend.flatten(y_pred[:,:,:,index])
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
        dice += (intersection + smooth) / (union + smooth)
    return -2./num_classes * dice

# Parameter
image_size = (224, 224)
model_path = os.path.join("model/model-unet.h5")
video_path = os.path.join("b4-converted.mp4-00.07.44.544-00.08.07.367.mp4")

# Load model
model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss, 'MaxMeanIoU': MaxMeanIoU})

# Load Video
cap = cv2.VideoCapture(video_path)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        image_height = frame.shape[0]
        image_width = frame.shape[1]
        frame = frame[0:image_height, (image_width-image_height)//2:(image_width-image_height)//2+image_height]
        frame_ori = frame.copy()
        frame = cv2.resize(frame, image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

        # Predict mask
        pred = model.predict(np.expand_dims(frame, 0))

        # Process mask
        mask = pred.squeeze()
        mask = np.stack((mask,)*3, axis=-1)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        mask_land = mask[:, :, 0]
        mask_sky = mask[:, :, 1]
        
        # Post Process
        mask_land = cv2.cvtColor(mask_land, cv2.COLOR_BGR2GRAY)
        mask_sky = cv2.cvtColor(mask_sky, cv2.COLOR_BGR2GRAY)

        border = get_border(mask_land, mask_sky)
        m, c = get_horizon_line(border, True)

        resized_image_height = frame.shape[0]
        resized_image_width = frame.shape[1]
        roll, pitch = get_roll_pitch(m, c, resized_image_height, resized_image_width)

        if mask_land[0,0]==1 or mask_land[0,224]==1:
            if roll > 0:
                roll = -180 + roll
            else:
                roll = 180 + roll

        #frame_ori = cv2.resize(frame_ori, (480, 480))
        scale = image_height/image_size[0]
        cv2.imshow("Original", frame_ori)

        frame_ori = draw_horizon_line(frame_ori, m, c, scale, (125, 0, 255))

        text_roll = "roll:" + str(round(roll, 2)) + " degree"
        text_pitch = "pitch:" + str(round(pitch, 2)) + " %"

        cv2.putText(frame_ori, "RANSAC :", (5, 15), 0, 0.5, (125, 0, 255), 2)
        cv2.putText(frame_ori, text_roll, (5, 35), 0, 0.5, (125, 0, 255), 2)
        cv2.putText(frame_ori, text_pitch, (5, 55), 0, 0.5, (125, 0, 255), 2)

        ###################################################################################

        m, c = get_horizon_line(border)

        resized_image_height = frame.shape[0]
        resized_image_width = frame.shape[1]
        roll, pitch = get_roll_pitch(m, c, resized_image_height, resized_image_width)

        #frame_ori = cv2.resize(frame_ori, (480, 480))
        scale = image_height/image_size[0]

        frame_ori = draw_horizon_line(frame_ori, m, c, scale, (255, 0, 125))

        text_roll = "roll:" + str(round(roll, 2)) + " degree"
        text_pitch = "pitch:" + str(round(pitch, 2)) + " %"

        cv2.putText(frame_ori, "LinReg :", (5, 95), 0, 0.5, (255, 0, 125), 2)
        cv2.putText(frame_ori, text_roll, (5, 115), 0, 0.5, (255, 0, 125), 2)
        cv2.putText(frame_ori, text_pitch, (5, 135), 0, 0.5, (255, 0, 125), 2)

        ###################################################################################

        cv2.imshow("Horizon", frame_ori)
        cv2.imshow("Land", mask_land)
        cv2.imshow("Border", border)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

print("Video Ended")
cap.release()

cv2.destroyAllWindows()
