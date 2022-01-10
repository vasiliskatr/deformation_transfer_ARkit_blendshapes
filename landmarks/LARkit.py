import numpy as np
LM = np.zeros(15).astype(int)
### Indices for key landmarks
LM[0] = 360 #lm_nose_tip = 30 

LM[1] = 1191 #lm_right_eye_R = 45 
LM[2] =  1203#lm_right_eye_L = 42

LM[3] =  606#lm_left_eye_L = 36 
LM[4] =  436#lm_left_eye_R = 39

LM[5] =  140#lm_mouth_L = 48
LM[6] =  824#lm_mouth_R = 54

LM[7] =  223#lm_mouth_middle_Top_T = 51
LM[8] = 776#lm_mouth_middle_Top_B = 62

LM[9] =  767#lm_mouth_middle_Bottom_T = 66
LM[10] =  242#lm_mouth_middle_Bottom_B = 57


LM [11] =  546# boundary_left
LM [12] =  1162 # boundary_right
LM [13] =  509# boundary_left_top
LM [14] =  1080# boundary_right_top

