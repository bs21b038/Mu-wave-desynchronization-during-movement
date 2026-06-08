from math import atan,cos,sin,pi,acos,exp
import numpy as np
from scipy import spatial
import random

import matplotlib.pyplot as plt
import pickle

from pathlib import Path

ROOT = Path(__file__).resolve().parent

def get_angle_wrt_x_axis(pt1,pt2):
    ang = abs(atan((pt2[1]-pt1[1])/(pt2[0]-pt1[0])))
    
    if pt1[0]<pt2[0] and pt1[1]<pt2[1]: #1st quadrant
        return ang
    elif pt1[0]>pt2[0] and pt1[1]<pt2[1]: #2nd quadrant
        return pi-ang
    elif pt1[0]>pt2[0] and pt1[1]>pt2[1]: #3rd quadrant
        return pi+ang    
    elif pt1[0]<pt2[0] and pt1[1]>pt2[1]: #4th quadrant
        return 2*pi-ang
    
def forward_left(MN):
    theta_s = ((MN[2]-MN[3])*pi/2)+pi/2
    theta_e = ((MN[0]-MN[1])*pi/2)+pi/2
    # x = (L_s)*cos(theta_s)+L_e*cos(theta_s-theta_e) - 0.15
    # y = (L_s)*sin(theta_s)+L_e*sin(theta_s-theta_e)
    x = (L_s+L_e*cos(theta_e))*cos(theta_s) + L_e*sin(theta_e)*sin(theta_s) - 0.15
    y = (L_s+L_e*cos(theta_e))*sin(theta_s) - L_e*sin(theta_e)*cos(theta_s)
    
    return x,y

def forward_right(MN):
    theta_s = ((MN[2]-MN[3])*pi/2)+pi/2
    theta_e = ((MN[0]-MN[1])*pi/2)+pi/2
    x = (-L_s-L_e*cos(theta_e))*cos(theta_s) - L_e*sin(theta_e)*sin(theta_s) + 0.15
    y = -(-L_s-L_e*cos(theta_e))*sin(theta_s) - L_e*sin(theta_e)*cos(theta_s)
    # x = -(L_s)*cos(theta_s)-L_e*cos(theta_s-theta_e) + 0.15
    # y = (L_s)*sin(theta_s)+L_e*sin(theta_s-theta_e)
    
    return x,y

def find_centre(x1,y1,x2,y2,r):
    xdiff = x1-x2
    ydiff = y1-y2
    s = x1**2 - x2**2 + y1**2 - y2**2
    Cy = (s/(2*ydiff)) - y1
    A = (1+((xdiff/ydiff)**2))
    B = -2*(((x1*ydiff)+(Cy*xdiff))/ydiff)
    C = x1**2 + Cy**2 - r**2
    cx_1 = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
    cx_2 = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
    cy_1 = (s/(2*ydiff)) - ((xdiff/ydiff)*cx_1)
    cy_2 = (s/(2*ydiff)) - ((xdiff/ydiff)*cx_2)

    return cx_1,cy_1,cx_2,cy_2

def get_x_y_components_spline(pt1,pt2,tang):
    th = get_angle_wrt_x_axis(pt1,pt2)
    ang = pi/2-th
    x_fac = cos(ang)*tang
    y_fac = sin(ang)*tang
    
    return x_fac,y_fac

def get_circle_params(pt1,pt2,tang):
    d = spatial.distance.euclidean(pt1,pt2)
    r = (((d**2)/(4*tang))+tang)/2
    c = find_centre(pt1[0],pt1[1],pt2[0],pt2[1],r)
    ang = acos((r-tang)/r)
    
    return c, r, 2*ang

# Load distance vs time
with open(ROOT / "time_vs_dist.txt", "rb") as fp:
    d_vs_t = pickle.load(fp)
    d_vs_t = np.array(d_vs_t)
    time_perc = d_vs_t[:,0]
    dist_perc = d_vs_t[:,1]
    
print("First 10 distance percentages:")
print(dist_perc[:10])

print("\nLast 10 distance percentages:")
print(dist_perc[-10:])

# Initialize parameters
arm_left = [-0.15,0]    
arm_right = [0.15,0]
L_s = 0.3
L_e = 0.3
peak_velocity_def  = 0.03
distance_def = 0.6
curvature = 0.07

# Define targets
targets = []
X = list(np.linspace(-0.7,0.7,29))
Y = list(np.linspace(0.31,0.6,29))
for x in X:
    for y in Y:
        targets.append([x,y])

# Get deducible parameters 
# Home position
home_left = forward_left([0.8,0.2,0.8,0.2])
home_right = forward_right([0.8,0.2,0.8,0.2])

print("home_left =", home_left)
print("home_right =", home_right)

# Output array for all targets
output = []
isochrony = []
n = 0
for target in targets:
    if target[0]<0:
        move_flag = "L"
    else:
        move_flag = "R"
        
    if move_flag=="L":
        # Initialize angle and arc parameters
        dist = spatial.distance.euclidean(home_left,target) # Distance between home and target
        if spatial.distance.euclidean(arm_left,target)>0.57:
            continue # Not reachable
        perp_max = curvature*dist # Maximum perpendicular distance of splined trajectory
        cent,rad,theta = get_circle_params(home_left,target,perp_max)
        cent = cent[2:] # Consider corresponding center
        theta_ch = get_angle_wrt_x_axis(cent,home_left) # Angle of center and home
        theta_ct = get_angle_wrt_x_axis(cent,target) # Angle of center and target
                
        # Initialize end effectors array
        # end_effectors = [np.array([home_left[0],home_left[1],home_right[0],home_right[1]])] # Starting movement from home position
        end_effectors = []
        for d in dist_perc:       
            theta_cov = d/100*theta
            if target[1]<home_left[1]:
                x_t = rad*cos(theta_ch-theta_cov)+cent[0] # Get x coordinate of resulting position
                y_t = rad*sin(theta_ch-theta_cov)+cent[1] # Get y coordinate of resulting position
            else:
                x_t = rad*cos(theta_ch+theta_cov)+cent[0] # Get x coordinate of resulting position
                y_t = rad*sin(theta_ch+theta_cov)+cent[1] # Get y coordinate of resulting position
            end_effectors.append(np.array([x_t,y_t,home_right[0],home_right[1]])) # Save to array           
        # end_effectors.append(np.array([target[0],target[1],home_right[0],home_right[1]])) # End movement with target location
        end_effectors = np.array(end_effectors)
        
    if move_flag=="R":
        # Initialize angle and arc parameters
        dist = spatial.distance.euclidean(home_right,target) # Distance between home and target
        if spatial.distance.euclidean(arm_right,target)>0.57:
            continue # Not reachable
        perp_max = curvature*dist # Maximum perpendicular distance of splined trajectory
        cent,rad,theta = get_circle_params(home_right,target,perp_max)        
        cent = cent[:2] # Consider corresponding center
        theta_ch = get_angle_wrt_x_axis(cent,home_right) # Angle of center and home
        theta_ct = get_angle_wrt_x_axis(cent,target) # Angle of center and target
      
        # Initialize end effectors array
        # end_effectors = [np.array([home_left[0],home_left[1],home_right[0],home_right[1]])] # Starting movement from home position
        end_effectors = []
        for d in dist_perc:       
            theta_cov = d/100*theta
            if target[1]<home_right[1]:
                x_t = rad*cos(theta_ch+theta_cov)+cent[0] # Get x coordinate of resulting position
                y_t = rad*sin(theta_ch+theta_cov)+cent[1] # Get y coordinate of resulting position
            else:
                x_t = rad*cos(theta_ch-theta_cov)+cent[0] # Get x coordinate of resulting position
                y_t = rad*sin(theta_ch-theta_cov)+cent[1] # Get y coordinate of resulting position
            end_effectors.append(np.array([home_left[0],home_left[1],x_t,y_t]))
        # end_effectors.append(np.array([home_left[0],home_left[1],target[0],target[1]])) # End movement with target location
        end_effectors = np.array(end_effectors)

    output.append(end_effectors)
    
with open(
    ROOT.parent.parent
    / "data"
    / "raw"
    / "splined_trajectories_3.txt",
    "wb"
) as fp:
    pickle.dump(output, fp)
    