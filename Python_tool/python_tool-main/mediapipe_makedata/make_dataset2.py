import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['GOOD']   #하고자 하는 동작 클래스 
#seq_length = 30
secs_for_action = 3   #찍고자 하는 동작 시간  

# MediaPipe hands model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
count =0
flag =0
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
        
    while cap.isOpened():
        for idx, action in enumerate(actions):
            LH_data = []
            RH_data = []
            POSE_data =[]
            POSE_HAND_data = []
            ret, img = cap.read()

            img = cv2.flip(img, 1)

            cv2.putText(img, f'Waiting for 3 seconds for next [{action.upper()}] action ...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)
            cv2.imshow('img',img)
            cv2.waitKey(3000)

            start_time = time.time()

            while time.time() - start_time < secs_for_action:
                ret, img = cap.read()

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                results = holistic.process(img)
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                mp_drawing.draw_landmarks(
                    img,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                
                #########################################################
                ###########               오른손              ###########
                #########################################################
                if results.left_hand_landmarks is not None:
                    cnt = 0
                    #print(cnt)
                    joint = np.zeros((21, 3))

                    for res in results.left_hand_landmarks.landmark:           
                        joint[cnt] = [res.x,res.y,res.z]
                        cnt+=1
                    
                    #print(joint)

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    #angle_label = np.append(angle_label, idx)
                    
                    
                    d = np.concatenate([joint.flatten(), angle_label.flatten()])
                    #print(d)

                    #LH_data.append(d)
                    
                    #print(cnt)
                
                else:
                    
                    joint = np.ones((21, 3))
                    
                    #print(joint)

                        # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    #angle_label = np.append(angle_label, idx)
                    

                    d = np.concatenate([joint.flatten(), angle_label.flatten()])

                    #LH_data.append(d)
                #########################################################
                #########################################################
                #########################################################


                

                #########################################################
                ############               왼손              ############
                #########################################################
                # if results.right_hand_landmarks is not None:
                #     cnt = 0

                #     #print(cnt)

                #     joint = np.zeros((21, 3))
                #     for res in results.right_hand_landmarks.landmark:
                        
                #         joint[cnt] = [res.x,res.y,res.z]
                #         cnt+=1
                    
                #     #print(joint)

                #         # Compute angles between joints
                #     v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                #     v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                #     v = v2 - v1 # [20, 3]
                #     # Normalize v
                #     v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                #     # Get angle using arcos of dot product
                #     angle = np.arccos(np.einsum('nt,nt->n',
                #         v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                #         v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                #     angle = np.degrees(angle) # Convert radian to degree

                #     angle_label = np.array([angle], dtype=np.float32)
                #     angle_label = np.append(angle_label, idx)

                #     d = np.concatenate([joint.flatten(), angle_label])

                #     RH_data.append(d)
                   
                #     #print(cnt)
                
                # else:
                    
                #     joint = np.ones((21, 3))
                    
                #     #print(joint)

                #     # Compute angles between joints
                #     v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                #     v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                #     v = v2 - v1 # [20, 3]
                #     # Normalize v
                #     v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                #     # Get angle using arcos of dot product
                #     angle = np.arccos(np.einsum('nt,nt->n',
                #         v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                #         v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                #     angle = np.degrees(angle) # Convert radian to degree

                #     angle_label = np.array([angle], dtype=np.float32)
                #     angle_label = np.append(angle_label, idx)

                #     d = np.concatenate([joint.flatten(), angle_label])

                #     RH_data.append(d)
                #########################################################
                #########################################################
                #########################################################



                #########################################################
                ############             어깨,팔             ############
                #########################################################
                if results.pose_landmarks is not None:
                    cnt = 0

                    #print(cnt)

                    joint = np.zeros((33, 3))
                    for res in results.pose_landmarks.landmark:
                        
                        joint[cnt] = [res.x,res.y,res.z]
                        cnt+=1
                    


                    # Compute angles between joints
                    v1 = joint[[11,11,13, 12,14], :3] # Parent joint
                    v2 = joint[[12,13,15, 14,16], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',   #백터별 구부러진 각도 계산 cos 이용
                        v[[0,1,0,3],:],  
                        v[[1,2,3,4],:])) 

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    #angle_label = np.append(angle_label, idx)
                    joint = joint[11:16]
                    e = np.concatenate([joint.flatten(), angle_label.flatten()])

                    #POSE_data.append(d)
                   
                    #print(cnt)
                
                else:
                    
                    joint = np.ones((33, 3))
                    
                    #print(joint)

                    # Compute angles between joints
                    v1 = joint[[11,11,13, 12,14], :3] # Parent joint
                    v2 = joint[[12,13,15, 14,16], :3] # Child joint
                    v = v2 - v1 # [20, 3]      
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] ##길이를 1로 정규화(?)해줌

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',   #백터별 구부러진 각도 계산 cos 이용
                        v[[0,1,0,3],:],  
                        v[[1,2,3,4],:])) 
# angle 설명:
# 벡터 a 와 벡터 b 를 내적(dot product)하면 [벡터 a의 크기] × [벡터 b의 크기] × [두 벡터가 이루는 각의 cos값] 이 됩니다.
# 그런데 바로 위에서 벡터들의 크기를 모두 1로 표준화시켰으므로 두 벡터의 내적값은 곧 [두 벡터가 이루는 각의 cos값]이 됩니다.
# 따라서 이것을 코사인 역함수인 arccos에 대입하면 두 벡터가 이루는 각이 나오게 됩니다.

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    #angle_label = np.append(angle_label, idx)
                    joint = joint[11:16]
                    e = np.concatenate([joint.flatten(), angle_label.flatten()])

                    #POSE_data.append(d)
                #########################################################
                #########################################################
                #########################################################

                #여기
                f = np.concatenate([d.flatten(), e.flatten()])
                POSE_HAND_data.append(f)


                cv2.imshow('img', img)   
                if cv2.waitKey(5) & 0xFF == 27: ####### esc 누르면 종료
                    exit(0)

                
            
                     
            #LH_data = np.array(LH_data)
            #RH_data = np.array(RH_data)
            #POSE_data = np.array(POSE_data)
            #print(action, LH_data.shape)
            #print(action, RH_data.shape)
            #print(action, POSE_data.shape)
            # POSE_HAND_data = np.array(POSE_HAND_data)
            print(len(POSE_HAND_data))

            RESULT = []
            #RESULT.append(LH_data)
            # RESULT.append(POSE_data)
            RESULT.append(POSE_HAND_data)
            RESULT.append(idx)
            RESULT = np.array(RESULT)
            print(RESULT.shape)

            ###############데이터셋 저장################
            # np.save(os.path.join('dataset', f'{action}_{count}_RH'), LH_data)
            # #np.save(os.path.join('dataset', f'{action}_{count}_LH'), RH_data)
            # np.save(os.path.join('dataset', f'{action}_{count}_POSE'), POSE_data)
            np.save(os.path.join('dataset', f'RESULT_{action}_{count}'), RESULT)
            if(flag==2):
                count+=1
                flag=0
            else:
                flag +=1
            ############################################


            # #Create sequence data
            # full_seq_data = []
            # for seq in range(len(data) - seq_length):
            #     full_seq_data.append(data[seq:seq + seq_length])

            # full_seq_data = np.array(full_seq_data)
            # print(action, full_seq_data.shape)
            # np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
        