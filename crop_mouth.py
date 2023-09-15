#! /usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#
#   file name: crop_mouth.py
#   author: tripletone
#   email: yangleisxn@gmail.com
#   created date: 2023/09/15
#   description: crop mouth from face images
#
#================================================================

import cv2
import dlib
import argparse
import numpy as np

lip_size_limit = 128
out_shape = (128, 64)

def mouth(img, shape, predictor, detector):
    # get faces
    faces = detector(img, 1)
    if len(faces) != 1:
        cv2.imwrite("no_face.jpg", img)
        return None

    # get mouth
    face = faces[0]
    points = predictor(img, face)
    mouth_points = [(point.x, point.y) for point in points.parts()[48:]]
    
    center = np.mean(mouth_points, axis=0).astype(int)
    rect = cv2.boundingRect(np.array(mouth_points))
    print(f"mouth in pics is {rect[2]}x{rect[3]}", end='...')
    
    if rect[2] < lip_size_limit:
        print('too small')
        return None
    else:
        shape = (max(shape[0], int(rect[2] / 0.6)),
                 max(shape[1], int(rect[3] / 0.6)))
        print(f"crop to {shape[0]}x{shape[1]}")

    return img[int(center[1] - shape[1]/2):int(center[1] + shape[1]/2),
               int(center[0] - shape[0]/2):int(center[0] + shape[0]/2)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop mouth from face')
    parser.add_argument('--input', type=str, default='face.jpg', 
                        help='Filenames of input images', required=True)
    parser.add_argument('--output', type=str, default='mouth.jpg',
                        help='Filenames of output images')
    args = parser.parse_args()

    predictor_path = 'weights/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    image = cv2.imread(args.input, cv2.IMREAD_COLOR)

    cropped_mouth = mouth(image,
                          shape=out_shape,
                          predictor=predictor,
                          detector=detector)
    cv2.imwrite(args.output, cropped_mouth)
