import Box2D as b2
from stable_baselines.common.vec_env import DummyVecEnv
import gym
import AerialManipulatorDRL
import numpy as np
import matplotlib.pyplot as plt



def PDpolicy_Arm_Trac_1_DOF(obs,f_max,step,trajck_x,trajck_y,track_beta,track_vx,track_vy, PD_params = None):
    #print(obs)
    obs = obs[0]
    pos = np.array([obs[0], obs[1]])
    vel = np.array([obs[8], obs[9]])
    ang = obs[10]
    avel = obs[11]
    tailAngle = obs[15]

   
    desPos = np.array([trajck_x[step], trajck_y[step]])
    desVel = np.array([track_vx[step], track_vy[step]])
    destailAngle = track_beta[step]
    gripmode = obs[11]



    epTail = destailAngle -tailAngle
    ep = desPos - pos
    ev = desVel - vel

    
    
    Kp_vert, Kd_vert, Kp_hor, Kd_hor, Kp_ang, Kd_ang,Kp_manip = PD_params

    armCommand = (destailAngle -tailAngle) *Kp_manip #for non beta trajcrectory test remove this 
    f_vert = Kp_vert * ep[1] + Kd_vert * ev[1] + 0.8 * 9.81
    f = f_vert/np.cos(ang)

    f_hor = Kp_hor * ep[0] + Kd_hor * ev[0]
    desAng = -np.arctan2(f_hor,f_vert)
    angleLimit = 1
    if desAng > angleLimit:
        desAng = angleLimit
    if desAng < -angleLimit:
        desAng = -angleLimit
    ealpha = desAng - ang
    ew = - avel

    torq = Kp_ang * ealpha + Kd_ang * ew
    if f > f_max:
        f = f_max
    if f < 0:
        f = 0


    #written for non beta tracjectory tests
    if (np.abs(pos[0]) <= 0.1 and  np.abs(pos[1]) <= 0.1  ):
        fingerSpeed =1;
        fingerSpeed =1;
    else:
        fingerSpeed = 0;


    
    return np.array([[(f/2-torq/2)* 4 / f_max-1, (f/2+torq/2)* 4 / f_max-1,fingerSpeed,armCommand*4]]) , desAng


