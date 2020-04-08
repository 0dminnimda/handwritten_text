import numpy as np
import cv2 as cv

def gtbp(name, nam="Tracking"):
    out_name = []
    for i in name:
        out_name.append(cv.getTrackbarPos(i, nam))
    return out_name

def stbp(name, nam="Tracking"):
    for i, j in name.items():
        cv.setTrackbarPos(i, nam, int(j))

def ctb(name, val, max=255, nam="Tracking"):
    for i in range(len(name)):
        cv.createTrackbar(name[i], nam, val[i], max, lambda _:_)

def grid_list():
    grid_img[:, :, :] = (255, 255, 255)
    for i in range(w//wi):
        for j in range(h//hi):
            if i == 2:
                col = RED
            else:
                col = GRAY
            
            cv.line(grid_img, (0, j*hi+hof), (w, j*hi+hof), GRAY, 1)
            cv.line(grid_img, (i*wi+wof, 0), (i*wi+wof, h), col, 1)

# img import and data preparation
img = cv.imread('list_055425_080420.png')
h, w, _ = img.shape

grid_img = np.zeros_like(img, np.uint8)

GRAY = (127, 127, 127)
GRAY = (127*0.5, 127*0.5, 127*0.5)
RED = (0, 0, 200)

# trackbar
for _ in range(1):
    cv.namedWindow("Tracking", cv.WINDOW_NORMAL)
    name1 = ["LH", "LS", "LV", "UH", "US", "UV"]
    ctb(name1, [105, 60, 123, 135, 255, 189])

    l_b, u_b = np.array_split(gtbp(name1), 2)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, l_b, u_b)
    cv.destroyAllWindows()

    cv.namedWindow("Tracking2", cv.WINDOW_NORMAL)
    name1 = ["mi", "wof", "hof"]
    ctb(name1, [48, 11, 1], max=100, nam="Tracking2")

    mi, wof, hof = gtbp(name1, nam="Tracking2")

#wof = 6
#hof = 17
#l_b, u_b = np.array(n[0:3]), np.array(n[3:6])
#mask = cv.inRange(hsv, l_b, u_b)

one_st = True

inv_mask = cv.bitwise_not(mask)

while 1: 
    mi, wof, hof = gtbp(name1, nam="Tracking2")
    wi, hi = mi, mi
    #wi = w//wof
    #hi = h//hof

    grid_list()

    fg = cv.bitwise_or(img, img, mask=mask)
    #mask = cv.bitwise_not(mask)
    bk = cv.bitwise_or(grid_img, grid_img, mask=inv_mask)
    final = cv.bitwise_or(fg, bk)

    #cv.namedWindow("bet", cv.WINDOW_NORMAL)
    #cv.imshow('image', img)
    cv.imshow('final', final)

    if cv.waitKey(1) & 0xFF == ord('2'):
        cv.destroyAllWindows()
        break

    if one_st:
        break

cv.imwrite("some.png", final)