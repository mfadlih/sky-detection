import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import os
import math
import time
#import Protocol.Protocol as protocol

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

def get_horizon_line(border):
    """Get horizon line equation from border image"""
    # Get border data in x,y format
    y = np.argmax(border, axis=0)
    x = np.arange(len(y))
    border_data = np.vstack([x, y]).T

    # Remove 0 from border data
    border_data = border_data[border_data[:, -1] != 0]

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

def draw_horizon_line(img, m, c, scale):
    """Draw horizon line on image"""
    image_height = img.shape[0]
    image_width = img.shape[1]

    c = scale*c

    pt1 = (0, int(m*0+c))
    pt2 = (image_width, int(m*image_width+c))

    cv2.line(img, pt1, pt2, (125, 0, 255), 2)

    return img

# Parameter
image_size = (128, 128)
model_path = os.path.join("model/model_quant_edgetpu.tflite")
#video_path = os.path.join("prediction", "sample.mp4")
video_path = os.path.join("a4-converted.mp4-00.02.43.911-00.03.05.670.mp4")

# Load model
#interpreter = tflite.Interpreter(model_path=model_path)
interpreter = tflite.Interpreter(model_path,
  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Load Video
cap = cv2.VideoCapture(video_path)
while(cap.isOpened()):
    start_time = time.time()
    ret, frame = cap.read()
    if ret:
        image_height = frame.shape[0]
        image_width = frame.shape[1]
        frame = frame[0:image_height, (image_width-image_height)//2:(image_width-image_height)//2+image_height]
        frame_ori = frame.copy()
        frame = cv2.resize(frame, image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        frame = np.array(frame, dtype=np.float32)

        input_scale, input_zero_point = input_details["quantization"]
        frame = frame / input_scale + input_zero_point

        frame = np.expand_dims(frame, axis=0).astype(input_details["dtype"])

        # Predict mask
        interpreter.set_tensor(input_details["index"], frame)
        interpreter.invoke()

        # Process mask
        output_data = interpreter.get_tensor(output_details['index'])

        # Binary threshold probability
        output_data[output_data >= 127] = 255
        output_data[output_data < 127] = 0

        mask_land = output_data[0, :, :, 0]
        mask_sky = output_data[0, :, :, 1]
        
        # Post Process
        border = get_border(mask_land, mask_sky)
        m, c = get_horizon_line(border)

        resized_image_height = frame.shape[0]
        resized_image_width = frame.shape[1]
        roll, pitch = get_roll_pitch(m, c, resized_image_height, resized_image_width)

        if mask_land[0,0]==1 or mask_land[0,127]==1:
            if roll > 0:
                roll = -180 + roll
            else:
                roll = 180 + roll

        #protocol.send_attitude(roll, pitch)

        # FPS
        fps = 1.0/(time.time() - start_time)
        
        #frame_ori = cv2.resize(frame_ori, (480, 480))
        scale = image_height/image_size[0]
        frame_ori = draw_horizon_line(frame_ori, m, c, scale)

        text_roll = "roll:" + str(round(roll, 2)) + " degree"
        text_pitch = "pitch:" + str(round(pitch, 2)) + " %"
        text_fps = "fps:" + str(round(fps))

        cv2.putText(frame_ori, text_roll, (5, 15), 0, 0.5, (125, 0, 255), 2)
        cv2.putText(frame_ori, text_pitch, (5, 35), 0, 0.5, (125, 0, 255), 2)
        cv2.putText(frame_ori, text_fps, (5, 55), 0, 0.5, (125, 0, 255), 2)

        cv2.imshow("Horizon", frame_ori)
        cv2.imshow("Land", mask_land)
        cv2.imshow("Border", border)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

print("Video Ended")
cap.release()

cv2.destroyAllWindows()
