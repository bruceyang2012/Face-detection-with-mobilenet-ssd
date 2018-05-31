import numpy as np 

path = "dataset/"
annotation_file = "wider_face_train_bbx_gt.txt"
relative_img_path = "dataset/WIDER_train/images/"

face_bb = {}

with open(path + annotation_file) as f:
    l_filename = True
    l_bb_cnt = False
    l_bb = True
    l_bb =0 
    f_name = " "
    bb = []

    bb_cnt = -1

    for line in f:

        if(bb_cnt == 0):
            l_bb = False
            l_filename = True    
            bb_cnt = -1 

        if(l_filename):
            f_name = line[:-1]
            l_bb_cnt = True
            l_filename = False
            # print ("f_name : ", f_name)
            continue 

        if(l_bb_cnt):
            bb_cnt = int(line)
            bb_ref_cnt = bb_cnt
            l_bb = True
            l_bb_cnt = False 
            # print ("bb_cnt : ", bb_cnt)
            continue 

        if(l_bb): 

            if(bb_ref_cnt == bb_cnt):
                bb = [relative_img_path + f_name, relative_img_path + f_name, [300, 300]]
                # bb.append(bb_ref_cnt)

            if( bb_cnt > 0):

                bb_cnt -= 1
                bb_each = []
                bb_info = line.split(" ") # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose

                isValid = bb_info[7] # 0, 1 using only 0 
                isOccluded = bb_info[8] # 0,1,2 use only 0,1 
                isBlur = bb_info[4] # 0,1,2 use only 0,1  
                
                x1 = int(bb_info[0])
                x2 = x1 + int(bb_info[2])
                y1 = int(bb_info[1])
                y2 = y1 + int(bb_info[3])

                bb_each.append([x1,x2,y1,y2])
                class_id = 1 # face
                bb_each.append(class_id)
                
                if isValid==0 and isOccluded!=2 and isBlur!=2:
                    bb.append(bb_each)
                    # print ("bb: " , bb)
                    # print ("line : ", line)

            if(bb_cnt == 0):
                face_bb[f_name] = bb 

            
#print (face_bb)
np.save('wider_test.npy', face_bb)
