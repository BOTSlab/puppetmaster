import cv2
from deeptag_model_setting import load_deeptag_models
from marker_dict_setting import load_marker_codebook
from stag_decode.detection_engine import DetectionEngine 
from util.homo_transform import warpPerspectivePts
import json
import os
import pprint
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw.draw import disk, line, polygon
from math import acos, cos, hypot, pi, sin
import matplotlib.pyplot as plt

guide_thumbnail_size = 41

def create_thumbnail():
    size = guide_thumbnail_size

    assert size % 2 != 0, 'guide_thumbnail_size should be odd'
    guide_thumbnail = np.zeros((size, size), np.uint8)

    # An arrow pointing to the right as a test for orientation
    rr, cc = line(size//2, 0, size//2, size//2)
    guide_thumbnail[rr, cc] = 255
    poly = np.array(( (size//4, size//2),
        (3*(size//4), size/2),
        (size//2, size-1)
    ))
    rr, cc = polygon(poly[:, 0], poly[:, 1])
    guide_thumbnail[rr, cc] = 255

    # Fill a disk in the centre.
    #rr, cc = disk((size//2, size//2), 10)
    #guide_thumbnail[rr, cc] = 255

    #gaussian_filter(guide_thumbnail, output=guide_thumbnail, sigma=5)

    max_value = np.amax(guide_thumbnail)
    guide_thumbnail = ((255 / max_value) * guide_thumbnail).astype(np.uint8)
    #print(guide_thumbnail)
    #plt.imshow(guide_thumbnail)
    #plt.show()

    return guide_thumbnail

def paste_thumbnail(centre, output_image, intensity=1.0):
    """Pastes the thumbnail image into output_image.  The tricky part is to
    handle the case where the pasted area overlaps the image boundary."""

    # Convert the centre to integer image indices.
    centre = list(map(int, centre))

    size = guide_thumbnail_size
    output_height = output_image.shape[0]
    output_width = output_image.shape[1]
    print(f"centre: {centre}")

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

    output_image[oy_start:oy_end, ox_start:ox_end] = intensity * guide_thumbnail[gy_start:gy_end, gx_start:gx_end]

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

        output_image = None
        cap = None 

        pp = pprint.PrettyPrinter(indent=4)

        while True:

            # read video frame or image
            if is_video:
                if cap is None: cap = cv2.VideoCapture(filename)
                ret, image =cap.read()
                if not ret:
                    print('Cannot read video.')
                    break
            else:
                image = cv2.imread(filename)
                if image is None:
                    print('Cannot read image.')
                    break

            if output_image is None:
                output_image = np.zeros((output_height, output_width), np.uint8)
            else:
                # output_image.fill(0)
                output_image = image[:,:,0] // 10

            # detect markers, print timing, visualize poses
            decoded_tags = stag_image_processor.process(image, detect_scale=None)
            #stag_image_processor.print_timming()

            # Write detected tag positions into the output image.
            for tag in decoded_tags:
                if not tag['is_valid']:
                    continue

                pp.pprint(tag)

                H = tag['H_crop']
                H_inv = np.linalg.inv(H)
                tag_centre = warpPerspectivePts(H_inv, [[127, 127]])[0]
                tag_right = warpPerspectivePts(H_inv, [[255, 127]])[0]

                paste_thumbnail(tag_centre, output_image)
                paste_thumbnail(tag_right, output_image, intensity=0.5)

                # To get the orientation of the tag, we use the positions of the tag's centre and a point directly to the right of the tag centre, as defined on the tag's "native" 256x256 grid.  Then we form two vectors, A and B and get the angle between them.  All of these quantities are defined in the image plane so there is an inherent assumption here that we have captured an overhead view.

                Ax = 1
                Ay = 0
                Bx = tag_right[0] - tag_centre[0]
                By = tag_right[1] - tag_centre[1]
                cos_theta = (Ax*Bx + Ay*By) / hypot(Bx, By)
                tag_theta = acos(cos_theta)
                #tag_theta = -pi/2

                # Determine the centre position for the thumbnail to be pasted.
                c = cos(tag_theta)
                s = sin(tag_theta)
                dx = c * guide_displacement[0] - s * guide_displacement[1]
                dy = s * guide_displacement[0] + s * guide_displacement[1]
                tag_centre[0] += dx
                tag_centre[1] += dy

                #paste_thumbnail(tag_centre, output_image)

            c = stag_image_processor.visualize(is_pause= not is_video)
            #cv2.imshow('Output Image', output_image)
            #c = cv2.waitKey(1)

            # press ESC or q to exit
            if c == 27 or c == ord('q') or not is_video:
                break
        

        if cap is not None: cap.release()
