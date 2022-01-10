import numpy as np
LM = np.zeros(18).astype(int)
### Indices for key landmarks
LM[0] = 5069 #lm_nose_tip 

LM[1] = 3888#lm_right_eye_outer 3888
LM[2] = 3621 #lm_right_eye_inner (tears) 3621

LM[3] = 1244  #lm_left_eye_inner (tears) 1244
LM[4] = 2023 #lm_left_eye_outer 2023

LM[5] = 6128#lm_mouth_R 
LM[6] = 5567  #lm_mouth_L 5567

LM[7] = 6323#lm_mouth_middle_Top_T 
LM[8] = 6248 #lm_mouth_middle_Top_B

LM[9] = 6399 #lm_mouth_middle_Bottom_T 
LM[10] = 6414 #lm_mouth_middle_Bottom_B 

LM[11] = 3138#center_chin

# Use edge vertices for reigid alignment (the less the better)
LM[12] =  2141 # top_of_mask_mid (edge vertice)
LM[13] =  1844 # mask_edge_Right_eyeheight (edge vertice)
LM[14] = 4063 # mask_edge_left_eyeheight (edge vertice)
LM[15] = 829# mask_edge_right_beloweyeheight (edge vertice)
LM[16] = 3083# mask_edge_left_beloweyeheight (edge vertice)
LM[17] = 2003# bottom_mask_neck)mid (edge vertice)



