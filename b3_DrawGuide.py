import cv2

from __config__ import *

# path_raw_resized = path_labeled+'/'+part

# print('path_raw_resized', path_raw_resized)
count = 0
for filename in os.listdir(path_raw_resized):
    count += 1
    # if filename[0] == 'r': # right
    #     route = 'phai'
    # elif filename[0] == 's': # straight
    #     route = 'thang'
    # elif filename[0] == 'l': # left
    #     route = 'trai'
    # else:
    #     route = None
    
    img_raw_path = path_raw_resized+'/'+filename
    print(img_raw_path, filename.split('__')[-1])

    img_raw = cv2.imread(img_raw_path)
    rh,rw = img_raw.shape[:2]
    # img_raw = img_raw[rh-HEIGHT:, :]
    # print('img_draw', img_draw.shape)
    
    if draw_guide_append:
        img_more = np.zeros(shape=[HEIGHT_MORE, 320, 3], dtype=np.uint8)
        if route is not None:
            if route == 'thang':
                cv2.circle(img_more, org_thang, HEIGHT_MORE//2, (0,0,255), -1)
            elif route == 'phai':
                cv2.circle(img_more, org_phai, HEIGHT_MORE//2, (0,0,255), -1)
            elif route == 'trai':
                cv2.circle(img_more, org_trai, HEIGHT_MORE//2, (0,0,255), -1)
            
            if lanepath == 'lanephai':
                cv2.rectangle(img_more, pts_lanephai[0], pts_lanephai[1], (255,0,0), -1)
            elif lanepath == 'lanetrai':
                cv2.rectangle(img_more, pts_lanetrai[0], pts_lanetrai[1], (255,0,0), -1)
            
            # if chuyenlane_to == 'trai': # chuyen sang trai
            #     cv2.rectangle(img_more, (90, 14), (40, 26), (0,255,0), -1)
            #     cv2.circle(img_more, (100, 20), 12, (0,255,0), -1)
            # elif chuyenlane_to == 'phai': # chuyen sang phai
            #     print('hiu', chuyenlane_to)
            #     # cv2.rectangle(img_more, (250, 30), (270, 40), (0,255,0), -1)
            #     # cv2.circle(img_more, (260, 16), 12, (0,255,0), -1)
            #     cv2.rectangle(img_more, (250, 14), (280, 26), (0,255,0), -1)
            #     cv2.circle(img_more, (240, 20), 12, (0,255,0), -1)
            
            if filename.split('__')[-1].split('.')[0] == 'sign': # khuc cua gap bien
                # if route == 'trai': # bien re trai
                #     cv2.rectangle(img_more, (90, 14), (40, 26), (0,255,0), -1)
                #     cv2.circle(img_more, (100, 20), 12, (0,255,0), -1)
                # else:
                #     cv2.rectangle(img_more, (250, 14), (280, 26), (0,255,0), -1)
                #     cv2.circle(img_more, (240, 20), 12, (0,255,0), -1)
                if route == 'trai': # bien re trai
                    cv2.rectangle(img_more, (HEIGHT_MORE, 15), (HEIGHT_MORE+60, HEIGHT_MORE-15), (0,255,0), -1)
                    cv2.circle(img_more, (HEIGHT_MORE+60, HEIGHT_MORE//2), HEIGHT_MORE//2-10, (0,255,0), -1)
                else:
                    cv2.rectangle(img_more, (250, 14), (280, 26), (0,255,0), -1)
                    cv2.circle(img_more, (240, HEIGHT_MORE//2), 12, (0,255,0), -1)


        img_draw = cv2.vconcat([img_raw, img_more])

    else:
        img_draw = img_raw.copy()

        if route is not None:
            if route == 'thang':
                cv2.circle(img_draw, org_thang, 20, (0,0,255), -1)
            elif route == 'phai':
                cv2.circle(img_draw, org_phai, 20, (0,0,255), -1)
            elif route == 'trai':
                cv2.circle(img_draw, org_trai, 20, (0,0,255), -1)
            
            if lanepath == 'lanephai':
                cv2.rectangle(img_draw, pts_lanephai[0], pts_lanephai[1], (255,0,0), -1)
            elif lanepath == 'lanetrai':
                cv2.rectangle(img_draw, pts_lanetrai[0], pts_lanetrai[1], (255,0,0), -1)
    
    cv2.imwrite(path_draw+filename, img_draw)
    if filename.split('__')[-1].split('.')[0] == 'sign':
        cv2.imshow('img_draw', img_draw)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

print('count', count)

cv2.destroyAllWindows()