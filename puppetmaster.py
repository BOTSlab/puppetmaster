import cv2
from deeptag_model_setting import load_deeptag_models
from marker_dict_setting import load_marker_codebook
from stag_decode.detection_engine import DetectionEngine 
import json
import os
import numpy as np
from scipy.ndimage import gaussian_filter


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config_video.json', help='path of configuration file for camera params and marker size')
    parser.add_argument('--CPU', action='store_true', help='use CPU only')
    args = parser.parse_args()


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

        # Create output image
        output_image = np.zeros((output_height, output_width), np.uint8)

        cap = None 

        while True:
            output_image.fill(0)

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

            # detect markers, print timing, visualize poses
            decoded_tags = stag_image_processor.process(image, detect_scale=None)
            #stag_image_processor.print_timming()

            # Write detected tag positions into the output image.
            for tag in decoded_tags:
                if not tag['is_valid']:
                    continue
                index_of_middle_kp = len(tag['keypoints_in_images']) // 2
                middle_kp = tag['keypoints_in_images'][index_of_middle_kp]
                output_image[int(middle_kp[1]), int(middle_kp[0])] = 255

            # Filter then normalize output image.
            # BAD: Close robots may have guide points that influence the other
            # robot when blurred.  Combine individual kernels through max filter???
            gaussian_filter(output_image, output=output_image, sigma=50)
            output_image = output_image / np.amax(output_image)

            cv2.imshow('Output Image', output_image)

            c = stag_image_processor.visualize(is_pause= not is_video)

            # press ESC or q to exit
            if c == 27 or c == ord('q') or not is_video:
                break
        

        if cap is not None: cap.release()

               
