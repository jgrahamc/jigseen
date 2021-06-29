# jigseen
#
# Copyright (c) 2021 John Graham-Cumming
#
# Small program to use SIFT algorithm to find likely location of a
# jigsaw piece

import cv2 as cv

# Pre-compute the SIFT parameters of the entire jigsaw

sift = cv.SIFT_create()
flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

jigsaw    = cv.imread('jigsaw.jpg')
jigsawg   = cv.cvtColor(jigsaw, cv.COLOR_BGR2GRAY)
jigsawrgb = cv.cvtColor(jigsaw, cv.COLOR_BGR2RGB)
kpj, desj = sift.detectAndCompute(jigsawg, None)

while True:

    # Capture an image of a jigsaw piece using the webcam by looping
    # until the user hits any key to indicate that the piece is in
    # place

    vid = cv.VideoCapture(0)
    piece = 0

    while True:
        ret, piece = vid.read()

        # Crop to middle of image. These parameters will need varying
        # for your webcam.
        
        piece = piece[400:640, 800:1120]
        cv.imshow('cam', piece)
      
        if cv.waitKey(1) != -1:
            break
  
    vid.release()
    cv.destroyWindow('cam')
    cv.waitKey(1) # Required to make the destroyWindow actually happen

    pieceg    = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
    kpp, desp = sift.detectAndCompute(pieceg, None)
    
    match = flann.knnMatch(desj, desp, k=2)
    mask = [[0,0] for i in range(len(match))]
    
    for i,(m,n) in enumerate(match):
        if m.distance < 0.6*n.distance:
            mask[i] = [1, 0]
            
    draw = dict(matchColor  = (0, 0, 0xFF),
                matchesMask = mask,
                flags       = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    output = cv.drawMatchesKnn(jigsaw, kpj, piece, kpp, match, None, **draw)

    # Show the resulting prediction of where the piece fits best until
    # the user hits a key

    cv.imshow('result', output)
    cv.waitKey(0)
    cv.destroyWindow('result')
    cv.waitKey(1) # See above
