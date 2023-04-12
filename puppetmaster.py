import cv2
from deeptag_model_setting import load_deeptag_models
from marker_dict_setting import load_marker_codebook
from stag_decode.detection_engine import DetectionEngine 
from stag_decode.visualization import visualize_rt
from util.homo_transform import warpPerspectivePts
import json
import os
import pprint
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, rotate
from skimage.draw.draw import disk, line, polygon
from math import atan2, cos, pi, sin
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon

from controller_set import ControllerSet

guide_thumbnail_method = "disk"
guide_thumbnail_size = 101
raw_window_name = "Raw Input"
input_window_name = "Input Image Side"
output_window_name = "Output Image Side"

cameraMatrix = None
distCoeffs = None
input_image = None
screen_corners = []
controller_set = ControllerSet()
green_or_blue = None

def create_thumbnail():
    size = guide_thumbnail_size

    assert size % 2 != 0, 'guide_thumbnail_size should be odd'

    if guide_thumbnail_method == "arrow":
    # An arrow pointing to the right as a test for orientation
        guide_thumbnail = np.zeros((size, size), np.uint8)
        rr, cc = line(size//2, 0, size//2, size//2)
        guide_thumbnail[rr, cc] = 255
        poly = np.array(( (size//4, size//2),
            (3*(size//4), size/2),
            (size//2, size-1)
        ))
        rr, cc = polygon(poly[:, 0], poly[:, 1])
        guide_thumbnail[rr, cc] = 255

    elif guide_thumbnail_method == "gaussian_disk":
        guide_thumbnail = np.zeros((size, size), np.uint8)
        rr, cc = disk((size//2, size//2), guide_thumbnail_size//3)
        guide_thumbnail[rr, cc] = 255
        gaussian_filter(guide_thumbnail, output=guide_thumbnail, sigma=10)

    elif guide_thumbnail_method == "distance":
        guide_thumbnail = np.ones((size, size), np.uint8)
        guide_thumbnail[size//2, size//2] = 0
        guide_thumbnail = distance_transform_edt(guide_thumbnail)
        guide_thumbnail = size//2 - guide_thumbnail
        guide_thumbnail[guide_thumbnail < 0] = 0
        max_value = np.amax(guide_thumbnail)
        guide_thumbnail = ((255 / max_value) * guide_thumbnail).astype(np.uint8)

    elif guide_thumbnail_method == "disk":
        guide_thumbnail = np.zeros((size, size), np.uint8)
        rr, cc = disk((size//2, size//2), size//2)
        guide_thumbnail[rr, cc] = 255

    #print(guide_thumbnail)
    #plt.imshow(guide_thumbnail)
    #plt.show()

    return guide_thumbnail

def paste_thumbnail(centre, rotation, output_image, intensity=1.0):
    """Pastes the thumbnail image into output_image.  The tricky part is to
    handle the case where the pasted area overlaps the image boundary."""

    # Convert the centre to integer image indices.
    centre = list(map(int, centre))

    size = guide_thumbnail_size
    output_height = output_image.shape[0]
    output_width = output_image.shape[1]
    #print(f"centre: {centre}")

    ox_start = centre[0] - size // 2
    oy_start = centre[1] - size // 2
    ox_end = ox_start + size
    oy_end = oy_start + size

    gx_start = 0
    gy_start = 0
    gx_end = size
    gy_end = size

    if ox_start < 0:
        gx_start = -ox_start
        ox_start = 0
    if oy_start < 0:
        gy_start = -oy_start
        oy_start = 0
    if ox_end >= output_width:
        gx_end -= ox_end - output_width
        ox_end = output_width - 1
    if oy_end >= output_height:
        gy_end -= oy_end - output_height
        oy_end = output_height - 1

    rotated_thumbnail = rotate(guide_thumbnail, -rotation*180/pi, reshape=False)

    output_image[oy_start:oy_end, ox_start:ox_end] = intensity * rotated_thumbnail[gy_start:gy_end, gx_start:gx_end]

def get_tag_theta(tag, img):
    rvecs = tag['rvecs']
    tvecs = tag['tvecs']

    x_vector = np.float32([[0.1,0,0]])
    origin = np.float32([[0,0,0]])

    x_vector_imgpts, jac = cv2.projectPoints(x_vector, rvecs, tvecs, cameraMatrix, distCoeffs)
    origin_imgpts, jac = cv2.projectPoints(origin, rvecs, tvecs, cameraMatrix, distCoeffs)

    #img = cv2.line(img, tuple(origin_imgpts[0].ravel().astype(np.int16).tolist()), \
    #              tuple(x_vector_imgpts[0].ravel().astype(np.int16).tolist()), (255, 255, 255), 5)

    # To get the orientation of the tag, we use the positions of the tag's centre and the tag's x-axis vector.  
    x = x_vector_imgpts[0][0][0] - origin_imgpts[0][0][0]
    y = x_vector_imgpts[0][0][1] - origin_imgpts[0][0][1]

    return atan2(y, x) + pi/2

def process_tag(tag, movement):
    forward_movement, angular_movement = movement

    H = tag['H_crop']
    H_inv = np.linalg.inv(H)
    tag_centre = warpPerspectivePts(H_inv, [[127, 127]])[0]

    theta = get_tag_theta(tag, output_image) + angular_movement

    # Movement vector relative to robot frame.
    Rx = guide_displacement[0] + forward_movement
    Ry = guide_displacement[1]

    # Rotate this vector into the world frame.
    c = cos(theta)
    s = sin(theta)
    Wx = c * Rx - s * Ry
    Wy = s * Rx + s * Ry
    tag_centre[0] += Wx
    tag_centre[1] += Wy

    paste_thumbnail(tag_centre, theta, output_image)

def draw_polygon(polygon, image, color):
    ring = polygon.exterior
    p1 = ring.coords[0]
    p1 = list(map(int, p1))
    for i in range(1, len(ring.coords)):
        p2 = ring.coords[i]
        p2 = list(map(int, p2))
        cv2.line(image, list(p1), p2, color, 5)
        p1 = p2

def visualize_input_side(image, decoded_tags):
    for decoded_tag in decoded_tags:
        if not decoded_tag['is_valid']: continue

        visualize_rt(image, decoded_tag['rvecs'], decoded_tag['tvecs'], cameraMatrix, distCoeffs, decoded_tag['tag_real_size_in_meter'], tag_id_decimal=decoded_tag['tag_id'], tid_text_pos=None, score=None, is_draw_cube =False, rotate_idx=0, text_color= None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json', help='path of configuration file for camera params and marker size')
    parser.add_argument('--CPU', action='store_true', help='use CPU only')
    args = parser.parse_args()

    guide_thumbnail = create_thumbnail()

    config_filename = args.config
    device = 'cpu' if args.CPU else None

    # load config
    load_config_flag = False
    try:
        config_dict = json.load(open(config_filename, 'r'))
        cameraMatrix = config_dict['cameraMatrix']
        distCoeffs = config_dict['distCoeffs']
        tag_real_size_in_meter = config_dict['marker_size']
        is_video = config_dict['is_video']!=0
        filename = config_dict['filepath']
        tag_family = config_dict['family']
        codebook_filename  = config_dict['codebook'] if len(config_dict['codebook']) else os.path.join('codebook', tag_family + '_codebook.txt')
        hamming_dist = config_dict['hamming_dist']
        output_width = config_dict['output_width']
        output_height = config_dict['output_height']
        screen_corners.append( config_dict['upper_left'] )
        screen_corners.append( config_dict['upper_right'] )
        screen_corners.append( config_dict['lower_right'] )
        screen_corners.append( config_dict['lower_left'] )
        guide_displacement = config_dict['guide_displacement']
        load_config_flag = True
    except:
        print('Cannot load config: %s'% config_filename)  

    # load models
    load_model_flag = False
    try:
        
        model_detector, model_decoder, device, tag_type, grid_size_cand_list = load_deeptag_models(tag_family, device) 
        load_model_flag = True
    except:
        print('Cannot load models.')

    # load marker library
    load_codebook_flag = False
    try:
        codebook = load_marker_codebook(codebook_filename, tag_type)
        load_codebook_flag = True
    except:
        print('Cannot load codebook: %s'% codebook_filename)

    # detection
    if load_config_flag and load_codebook_flag and load_model_flag:
        # initialize detection engine
        stag_image_processor = DetectionEngine(model_detector, model_decoder, device, tag_type, grid_size_cand_list, 
                    stg2_iter_num= 2, # 1 or 2
                    min_center_score=0.2, min_corner_score = 0.2, # 0.1 or 0.2 or 0.3
                    batch_size_stg2 = 4, # 1 or 2 or 4
                    hamming_dist= hamming_dist, # 0, 2, 4
                    cameraMatrix = cameraMatrix, distCoeffs=  distCoeffs, codebook = codebook,
                    tag_real_size_in_meter_dict = {-1:tag_real_size_in_meter})

        cameraMatrix = np.float32(cameraMatrix).reshape(3,3)
        distCoeffs = np.float32(distCoeffs)
        screen_corners_polygon = Polygon(screen_corners)

        output_corners = [[0, 0], [output_width-1, 0], [output_width-1, output_height-1], [0, output_height-1]]
        homography, status = cv2.findHomography(np.array(screen_corners), np.array(output_corners))

        #cv2.namedWindow(raw_window_name, cv2.WINDOW_NORMAL)
        #cv2.namedWindow(input_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(output_window_name, cv2.WINDOW_GUI_NORMAL)
        #cv2.setWindowProperty(raw_window_name, cv2.WND_PROP_TOPMOST, 1)
        #cv2.setWindowProperty(input_window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(output_window_name, cv2.WND_PROP_TOPMOST, 1)

        output_image = None
        cap = None 
        pp = pprint.PrettyPrinter(indent=4)

        while True:

            # read video frame or image
            if is_video:
                if cap is None: cap = cv2.VideoCapture(filename)
                ret, raw_image = cap.read()
                if not ret:
                    print('Cannot read video.')
                    break
            else:
                raw_image = cv2.imread(filename)
                if raw_image is None:
                    print('Cannot read image.')
                    break

            # Warp the raw image.
            #if input_image is None:
            #    input_image = np.zeros((output_height, output_width), np.uint8)
            input_image = cv2.warpPerspective(raw_image, homography, (output_width, output_height))

            if output_image is None:
                output_image = np.zeros((output_height, output_width), np.uint8)
            else:
                output_image.fill(0)
                # output_image = input_image[:,:,0] // 10

            # detect markers, print timing, visualize poses
            decoded_tags = stag_image_processor.process(input_image, detect_scale=None)
            #stag_image_processor.print_timming()

            # Compute a dictionary of the desired movements for all tags.  Tags
            # not corresponding to active robots will not have an entry.
            movement_dict = controller_set.get_movements(decoded_tags)

            # Write detected tag positions into the output image.
            for tag in decoded_tags:
                if not 'tag_id' in tag or not tag['tag_id'] in movement_dict:
                    # The first case is for invalid tags that don't have an id.
                    # The second covers valid tags that are not active robots,
                    # as determined by controller_set.
                    continue

                movement = movement_dict[tag['tag_id']]
                if abs(movement[0]) > 0 or abs(movement[1]) > 0:
                    # If the movement is non-zero, place a thumbnail on the screen.
                    # to guide the robot.
                    process_tag(tag, movement)

            #c = stag_image_processor.visualize(is_pause= not is_video)
            draw_polygon(screen_corners_polygon, raw_image, (255, 255, 0))
            visualize_input_side(input_image, decoded_tags)

            resize_divisor = 1
            if resize_divisor > 1:
                #cv2.resizeWindow(raw_window_name, raw_image.shape[1]//resize_divisor, raw_image.shape[0]//resize_divisor)
                #cv2.resizeWindow(input_window_name, input_image.shape[1]//resize_divisor, input_image.shape[0]//resize_divisor)
                cv2.resizeWindow(output_window_name, output_image.shape[1]//resize_divisor, output_image.shape[0]//resize_divisor)
            #cv2.imshow(raw_window_name, raw_image)
            #cv2.imshow(input_window_name, input_image)

            if green_or_blue is None:
                green_or_blue = np.zeros((output_height, output_width), np.uint8)
            rgb = cv2.merge([green_or_blue, green_or_blue, output_image])

            cv2.imshow(output_window_name, rgb)
            #cv2.imshow(output_window_name, output_image)
            c = cv2.waitKey(10)

            # press ESC or q to exit
            if c == 27 or c == ord('q') or not is_video:
                break

        if cap is not None: cap.release()
