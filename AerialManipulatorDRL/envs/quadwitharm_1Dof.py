import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from numpy import linalg as la
import Box2D as b2
import matplotlib.pyplot as plt

class myContactListener(b2.b2ContactListener):
    def __init__(self):
        b2.b2ContactListener.__init__(self)
        self.InContactLeft=False
        self.InContactRight=False
        self.IsGripped=False
        self.IsCrash =0
        self.IsArmPush = 0
    def BeginContact(self, contact):
        #print("in contact")
        bodyA=contact.fixtureA.body
        bodyB=contact.fixtureB.body
        worldManifold = contact.worldManifold

        if( ( bodyA.userData == 3 or bodyB.userData ==3 )  ):
            if(bodyA.userData == 3):
                if (bodyB.userData != 2):
                    self.IsCrash = 1
            elif( bodyB.userData == 3 ):
                if(bodyB.userData != 2):
                    self.IsCrash =1
        elif (bodyA.userData == 7 or bodyB.userData ==7 ):
            if (bodyA.userData == 7):
                points= self.GetContactPointX(bodyA,worldManifold)

            elif (bodyB.userData == 7):
                points= self.GetContactPointX(bodyB,worldManifold)

            #print(points, " valuse")
            if (points[0] < 0.02 ):
                self.InContactRight=True

        elif (bodyA.userData == 6 or bodyB.userData ==6 ):
            if (bodyA.userData == 6):
                points= self.GetContactPointX(bodyA,worldManifold)

            elif (bodyB.userData == 6):
                points= self.GetContactPointX(bodyB,worldManifold)

            if (points[0] > -0.02):
                self.InContactLeft=True
                #print("left contact")

            
        if (self.InContactLeft and self.InContactRight ):
            self.IsGripped=True
            #print("gripped")
        pass

    def EndContact(self, contact):
        #print("out contact")
        bodyA=contact.fixtureA.body
        bodyB=contact.fixtureB.body
        #print(bodyA.userData)
        #print(bodyB.userData)

        if( ( bodyA.userData == 3 or bodyB.userData ==3 )  ):
            if(bodyA.userData == 3):
                if (bodyB.userData != 2):
                    self.IsCrash = 0
            elif( bodyB.userData == 3 ):
                if(bodyB.userData != 2):
                    self.IsCrash =0
        elif(bodyA.userData == 6 or bodyB.userData ==6):
            if (self.InContactLeft==True):
                self.InContactLeft=False
        elif (bodyA.userData == 7 or bodyB.userData ==7 ):
            if(self.InContactRight==True):
                self.InContactRight=False



        if ((not self.InContactLeft) or (not self.InContactRight) ):
            self.IsGripped=False
        pass

    def PreSolve(self, contact, oldManifold):
        pass
    def PostSolve(self, contact, impulse):
        pass

    def GetContactPointX(self,body,CurrentManifold):
        point1 = CurrentManifold.points[0]
        point2 = CurrentManifold.points[1]
        local1 = body.GetLocalPoint(point1)
        local2 = body.GetLocalPoint(point2)
        return (local1[0],local2[0])
    def ResetValues(self):
        self.InContactLeft=False
        self.InContactRight=False
        self.IsGripped=False
        self.IsCrash =0
        self.IsArmPush =0 



class QuadWithArm1Dof(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
    #world values
        self.gravitational_acc = 9.81
        self.DEGTORAD= 0.0174533
        self.time_step = 1.0 / 100
        self.vel_iters, self.pos_iters = 1, 1
        self.ContactListener = myContactListener()
        self.priorKnowledgeChance = -0.1#0.1 # use this value to start the system with box gripped
        self.IsStartWithPrior = False
        self.IsRandomDesPosTrain = False # False # use this to start with random destination variable
        self.probPrior = np.random.uniform(low=0,high=1)

        self.countx =0
        self.axisValue = 2
        self.fingerStartAngle = 0#0.42
        self.fingerLimitAngle = 0.42
        self.tailLimitAngle  = 0.1
        self.tailStartAngle = 0#self.tailLimitAngle


    #box shape values
        self.boxStartPos = b2.b2Vec2(0.0,0.0)
        self.boxDistanceToOrigin = -0.25
        self.witdhBox = 0.125/2.0  #distance from center in local coordinates so total is this*2
        self.heightBox = 0.125/2.0 #distance from center in local coordinates
        self.posInitBox =  self.boxStartPos+b2.b2Vec2(0,self.boxDistanceToOrigin)    #b2.b2Vec2(0,-0.25)
        self.BoxuserData=2

    #static ground 1 values (contacts with box)
        self.witdhGround1 = 2*self.witdhBox #distance from center in local coordinates so total is this*2
        self.heightGround1 = 0.00
        self.posInitGround1 = self.posInitBox + b2.b2Vec2(0,-self.heightBox)

    #platform shap values
        self.witdhPlatform = 0.25 #distance from center in local coordinates so total is this*2
        self.heightPlatfrom = 0.01 #distance from center in local coordinates so total is this*2
        self.posInitPlatform = b2.b2Vec2(0,0)  # in reset function this changes with random
        self.lengthPlatform = self.witdhPlatform*2
        self.massPlatform = 0.4

	#tail shape values
        self.witdhTail = 0.013  #distance from center in local coordinates so total is this*2
        self.heightTail = 0.125/2 #distance from center in local coordinates so total is this*2
        self.lengthTail = 0.5
        self.massTail = 0.2
        

    #elbow values
        self.witdhElbow = 0.125/2 #distance from center in local coordinates so total is this*2
        self.heightElbow = 0.01 #distance from center in local coordinates so total is this*2
        self.lenghtElbow = self.witdhElbow*2
        self.massElbow = 0.05



    #finger values
        self.witdhFinger = 0.01  #distance from center in local coordinates so total is this*2
        self.heightFinger = 0.125/2 #distance from center in local coordinates so total is this*2
        self.lenghtFinger = self.heightFinger*2
        self.massFinger =0.05



        self.setPositions(self.posInitPlatform)




        
        self.world = b2.b2World(gravity=(0, -self.gravitational_acc),contactListener=self.ContactListener)  # initiate environment
        self.world.CreateDynamicBody(position=self.posInitPlatform,
                                    fixtures=b2.b2FixtureDef(shape=b2.b2PolygonShape(box=(self.witdhPlatform, self.heightPlatfrom)),
                                                            density=1,
                                                            friction=0),#quadrotor
                                    )
        self.world.CreateDynamicBody(position=self.posInitTail,
                                    fixtures=b2.b2FixtureDef(shape=b2.b2PolygonShape(box=(self.witdhTail,  self.heightTail)),
                                                            density=1,
                                                            friction=0),#tail
                                    )
        self.world.CreateDynamicBody(position=self.posInitBox,
                                    fixtures=b2.b2FixtureDef(shape=b2.b2PolygonShape(box=(self.witdhBox, self.heightBox)),
                                                            density=1,
                                                            friction=1), #box1
                                    )
        self.world.CreateStaticBody(position=self.posInitGround1,
                                    fixtures=b2.b2FixtureDef(shape=b2.b2EdgeShape(vertices=[(-self.witdhGround1,self.heightGround1),(self.witdhGround1,self.heightGround1)]),
                                                            density=1,
                                                            friction=1), #ground 1
                                    ) 

        self.world.CreateDynamicBody(position=self.posInitElbowLeft,
                                    fixtures=b2.b2FixtureDef(shape=b2.b2PolygonShape(box=(self.witdhElbow, self.heightElbow)),
                                                            density=1,
                                                            friction=0), #elbowleft
                                    )

        self.world.CreateDynamicBody(position=self.posInitElbowRight,
                                    fixtures=b2.b2FixtureDef(shape=b2.b2PolygonShape(box=(self.witdhElbow, self.heightElbow)),
                                                            density=1,
                                                            friction=0), #elbowright
                                    )

        self.world.CreateDynamicBody(position=self.posInitFingerLeft,
                                    fixtures=b2.b2FixtureDef(shape=b2.b2PolygonShape(box=(self.witdhFinger, self.heightFinger)),
                                                            density=1,
                                                            friction=1), #fingerleft
                                    )
        self.world.CreateDynamicBody(position=self.posInitFingerRight,
                                    fixtures=b2.b2FixtureDef(shape=b2.b2PolygonShape(box=(self.witdhFinger, self.heightFinger)),
                                                            density=1,
                                                            friction=1), #fingerright
                                    )


        bodybox = self.world.bodies[2]
        bodybox.angle = 0*self.DEGTORAD
        bodybox.mass = 0.01
        bodybox.inertia = 0.01 
        bodybox.userData=2     	

        ground1=self.world.bodies[3]
        ground1.userData=3

        
        self.thrust_to_weight_ratio = 2




        body = self.world.bodies[0]  # parameters of main body
        body.userData=0
        body.linearVelocity = b2.b2Vec2(0, 0)
        body.angle = 0
        body.angularVelocity = 0
        body.mass = self.massPlatform
        body.inertia = (1.0 * self.massPlatform * self.lengthPlatform * self.lengthPlatform) / 12.0
        #print("x",body.inertia)


        tail = self.world.bodies[1]  # parameters of tail
        tail.userData=1

        tail.linearVelocity = b2.b2Vec2(0, 0)
        tail.angle = 0
        tail.angularVelocity = 0
        tail.mass = self.massTail
        tail.inertia = (1.0 * self.massTail * self.lengthTail * self.lengthTail) / 12.0
        #print("xx",tail.inertia)


        elbowleft=self.world.bodies[4]
        elbowleft.userData=4
        elbowleft.mass = self.massElbow
        elbowleft.inertia =(1.0 * self.massElbow * self.lengthTail * self.lengthTail) / 12.0

        elbowright=self.world.bodies[5]
        elbowright.userData=5
        elbowright.mass=self.massElbow
        elbowright.inertia= (1.0 * self.massElbow * self.lengthTail * self.lengthTail) / 12.0

        fingerleft=self.world.bodies[6]
        fingerleft.userData=6
        fingerleft.mass=self.massFinger
        fingerleft.inertia=(1.0 * self.massFinger * self.lengthTail * self.lengthTail) / 12.0

        fingerright=self.world.bodies[7]
        fingerright.userData=7
        fingerright.mass=self.massFinger
        fingerright.inertia=(1.0 * self.massFinger * self.lengthTail * self.lengthTail) / 12.0
        #print("xxx",fingerright.inertia)

        self.totalMass=body.mass+tail.mass+elbowleft.mass+elbowright.mass+fingerleft.mass+fingerright.mass
        self.total_mass = self.totalMass
        self.f_0 = self.totalMass * self.gravitational_acc #hover force will be add as trim
        self.totalForce=  self.f_0 *  self.thrust_to_weight_ratio
        self.actionToForceConverter =  (self.totalForce- self.f_0)/2 # assuming action space is -1,1 this one motor force input 
        #print("f_0=",self.f_0,"  Total_foce:",self.totalForce,"   converter:",self.actionToForceConverter)


        self.massTorqueElbowLeft=fingerleft.mass*self.gravitational_acc*self.witdhElbow*2+elbowleft.mass*self.gravitational_acc*self.witdhElbow
        #print(self.massTorqueElbowLeft)
        tail_angle_limit = self.tailLimitAngle  # joint parameters
        elbow_tail_angle_limit = 0.0
        elbow_finger_angle_limit = self.fingerLimitAngle
        self.world.CreateRevoluteJoint(bodyA=body,
                                       bodyB=tail,
                                       localAnchorA=(0, 0),
                                       localAnchorB=(0, self.heightTail),
                                       enableLimit=True,
                                       lowerAngle=-tail_angle_limit,
                                       upperAngle=tail_angle_limit,
                                       motorSpeed=0.0,
                                       maxMotorTorque = 10.0, #max motor torque is set to keep the speed constant to if motor speed is zero it will act as a friction
                                       enableMotor=True,
                                       )

        self.world.CreateRevoluteJoint(bodyA=tail,
                                       bodyB=elbowleft,
                                       localAnchorA=(-self.witdhTail, -self.heightTail),
                                       localAnchorB=(self.witdhElbow, 0),
                                       enableLimit=True,
                                       lowerAngle=-elbow_tail_angle_limit,
                                       upperAngle=elbow_tail_angle_limit,
                                       motorSpeed=0,
                                       maxMotorTorque = 10.0, #2*elbowleft.mass*self.witdhElbow, #to simulate motor friction we set motor speed to zero but enabled motor itself to create a friction
                                       enableMotor=False,
                                       )

        self.world.CreateRevoluteJoint(bodyA=tail,
                                       bodyB=elbowright,
                                       localAnchorA=(self.witdhTail, -self.heightTail),
                                       localAnchorB=(-self.witdhElbow, 0),
                                       enableLimit=True,
                                       lowerAngle=-elbow_tail_angle_limit,
                                       upperAngle=elbow_tail_angle_limit,
                                       motorSpeed=0,
                                       maxMotorTorque = 10.0, # 2*elbowleft.mass*self.witdhElbow, #to simulate motor friction we set motor speed to zero but enabled motor itself to create a friction
                                       enableMotor=False,

                                       )
        self.world.CreateRevoluteJoint(bodyA=elbowleft,
                                       bodyB=fingerleft,
                                       localAnchorA=(-self.witdhElbow, 0),
                                       localAnchorB=(0, self.heightFinger),
                                       enableLimit=True,
                                       lowerAngle=-elbow_finger_angle_limit,
                                       upperAngle=elbow_finger_angle_limit,
                                       motorSpeed=0.0,
                                       maxMotorTorque = 1,
                                       enableMotor=True,
                                       )
        self.world.CreateRevoluteJoint(bodyA=elbowright,
                                       bodyB=fingerright,
                                       localAnchorA=(self.witdhElbow, 0.0),
                                       localAnchorB=(0.0, self.heightFinger),
                                       enableLimit=True,
                                       lowerAngle=-elbow_finger_angle_limit,
                                       upperAngle=elbow_finger_angle_limit,
                                       motorSpeed=0.0,
                                       maxMotorTorque = 1,
                                       enableMotor=True,
                                       )



        jointTailPLatform = self.world.joints[0]
        jointElbowleftFingerleft =self.world.joints[3]
        jointElbowrightFingerright =self.world.joints[4]

        vel = body.linearVelocity  # gym settings
        pos = body.position
        posBox =  bodybox.position #b2.b2Vec2(0,-0.25) #bodybox.position
        ang = body.angle
        avel = body.angularVelocity
        leftJointAngle= jointElbowleftFingerleft.angle
        rightJointAngle = jointElbowrightFingerright.angle
        tailJointAngle = jointTailPLatform.angle
        avelTail = jointTailPLatform.speed
        angleBox = bodybox.angle

        posFingerEndPoint = fingerleft.GetWorldPoint(b2.b2Vec2(self.witdhFinger, -self.heightFinger))
        posBoxFingerLeftPoint = bodybox.GetWorldPoint(b2.b2Vec2(-self.witdhBox, 0))
        #print (posFingerEndPoint)
        gripmode= 0;


        self.desPos = np.array([posBox[0],posBox[1]-self.boxDistanceToOrigin])  # destination
        self.desPosFinger = np.array([posBoxFingerLeftPoint[0],posBoxFingerLeftPoint[1]])  # destination finger
        self.relativePos = np.array([posBox[0],posBox[1]-self.boxDistanceToOrigin])   # use this with box start pos to change the start pos o box and use the value at step function with 3 pos observations
        #dont forget to change box out of order 

        #print("test",avelTail)
        self.desPosBoxDeterministic = np.array( [ -0.1, 0.1 ] )
        self.desPosBoxLow_x = -0.5
        self.desPosBoxHigh_x = 0.5

        self.desPosBoxLow_y =  0
        self.desPosBoxHigh_y = 0.5
        #easy fin pos
        self.desBoxPos = np.array([posBox[0]+ self.desPosBoxDeterministic[0],posBox[1]-self.boxDistanceToOrigin + self.desPosBoxDeterministic[1]])
        self.desBoxPosObs = b2.b2Vec2( self.desBoxPos[0], self.desBoxPos[1])



        if (self.IsRandomDesPosTrain ):

            self.prev_st = np.append([pos,posFingerEndPoint,posBox,self.desBoxPosObs, vel], [ang, avel,angleBox,gripmode,leftJointAngle,tailJointAngle]).reshape(16)
            self.st = np.append([pos,posFingerEndPoint,posBox,self.desBoxPosObs, vel], [ang, avel,angleBox,gripmode,leftJointAngle,tailJointAngle]).reshape(16)
            self.observation_space = spaces.Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0,-np.inf,-np.inf]),
                                            high=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf, np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1,np.inf,np.inf]),
                                            shape=None, dtype=np.float32)
        else:
            self.prev_st = np.append([pos,posFingerEndPoint,posBox, vel], [ang, avel,angleBox,gripmode,leftJointAngle,tailJointAngle]).reshape(14)
            self.st = np.append([pos,posFingerEndPoint,posBox, vel], [ang, avel,angleBox,gripmode,leftJointAngle,tailJointAngle]).reshape(14)
            self.observation_space = spaces.Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0,-np.inf,-np.inf]),
                                            high=np.array([np.inf,np.inf,np.inf,np.inf,np.inf, np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1,np.inf,np.inf]),
                                            shape=None, dtype=np.float32)


        self.act = np.array([0, 0, 0, 0])
        self.action_lim_high = 1.  
        self.action_lim_low = -1.
        self.action_space = spaces.Box(low=np.array([self.action_lim_low, self.action_lim_low,self.action_lim_low,self.action_lim_low]),
                                       high=np.array([self.action_lim_high, self.action_lim_high,self.action_lim_high,self.action_lim_high]), shape=None,
                                       dtype=np.float64)
        

        self.stepnum = 0
        self.step_per_episode = 400
        self.dn = False

        #print(self.st)

        self.motor_model_activated = False
        self.motor_speed_l = np.sqrt(1/self.thrust_to_weight_ratio)
        self.motor_speed_r = np.sqrt(1/self.thrust_to_weight_ratio)
        self.motor_settling_time = 0.15  # 0.02 settling time of motor, should be greater than 4*time_step

        #self.setDesPos()  # destination



        

        #hard fin pos
        #self.desBoxPos = np.array([-0.3,0.05])
        self.desVel = np.array([0, 0])
        self.desAng = np.array([0])
        self.desAvel = np.array([0])

        

        self.init_dist_pos_low_x = -1
        self.init_dist_pos_low_y = 0.2
        self.init_dist_pos_high_x = 1
        self.init_dist_pos_high_y = 1
        self.init_dist_vel = 0
        self.init_dist_ang = 0
        self.init_dist_avel = 0
        self.set_init_pos = None
        self.set_des_pos = None

        self.outOfBorderPos=4
        self.outOfBorderPosBox=4   #0.5



        self.hard_stop_episode = False
        self.hard_stop_enable = True

        self.OutOfBorder=False
        self.OutOfBorderBox=False
        self.CrashOccured = False 

        self.fg = None
        self.ax = None

    def step(self, action):

        body = self.world.bodies[0]
        tail = self.world.bodies[1]
        bodybox  = self.world.bodies[2]
        ground = self.world.bodies[3]
        elbowleft = self.world.bodies[4]
        elbowright = self.world.bodies[5]
        fingerleft = self.world.bodies[6]
        fingerright = self.world.bodies[7]

        jointtailbody = self.world.joints[0]
        jointTailElbowleft = self.world.joints[1]
        jointTailElbowright =self.world.joints[2]
        jointElbowleftFingerleft =self.world.joints[3]
        jointElbowrightFingerright =self.world.joints[4]

        #print(tail.angle,jointtailbody.angle)

        if type(action) == list:
            action = np.array(action, dtype=np.float64)
        self.act = action.astype(np.float64)
        motor_act = self.act.copy()


        """if (self.ContactListener.IsGripped):
            self.desPos[0]=0.0
            self.desPos[1]=0.2"""

        """if (self.ContactListener.IsGripped):
            print(jointElbowleftFingerleft.angle," ", jointElbowrightFingerright.angle )
            
            self.desPos[0]=0.5
            self.desPos[1]=0.5
            jointElbowleftFingerleft.motorSpeed= 7*(0.40-jointElbowleftFingerleft.angle)
            jointElbowrightFingerright.motorSpeed= 7*(-0.40-jointElbowrightFingerright.angle)
            extra_force=0.5
        else:
            if (np.abs(body.position[0]) <= 0.1 and  np.abs(body.position[1]) <= 0.1  and
                np.abs(body.linearVelocity[0]) <= 0.05 and  np.abs(body.linearVelocity[1]) <= 0.05  ):
                jointElbowleftFingerleft.motorSpeed= 4*(0.4-jointElbowleftFingerleft.angle)
                jointElbowrightFingerright.motorSpeed= 4*(-0.4-jointElbowrightFingerright.angle)
            else:
                jointElbowleftFingerleft.motorSpeed= 4*(-0.4-jointElbowleftFingerleft.angle)
                jointElbowrightFingerright.motorSpeed= 4*(0.4-jointElbowrightFingerright.angle)"""
                   
        """if (np.abs(body.position[0]) <= 0.1 and  np.abs(body.position[1]) <= 0.1  and
                np.abs(body.linearVelocity[0]) <= 0.05 and  np.abs(body.linearVelocity[1]) <= 0.05  ):
            self.desPos=[1,1]"""
        #print( motor_act[0],"  ", motor_act[1])

        if motor_act[0] > self.action_lim_high:  # clipping actions
            motor_act[0] = self.action_lim_high
        if motor_act[0] < self.action_lim_low:
            motor_act[0] = self.action_lim_low
        if motor_act[1] > self.action_lim_high:
            motor_act[1] = self.action_lim_high
        if motor_act[1] < self.action_lim_low:
            motor_act[1] = self.action_lim_low


        if motor_act[2] > self.action_lim_high:  # clipping actions
            motor_act[2] = self.action_lim_high
        if motor_act[2] < self.action_lim_low:
            motor_act[2] = self.action_lim_low

        """
        if motor_act[3] > self.action_lim_high:  # clipping actions
            motor_act[3] = self.action_lim_high
        if motor_act[3] < self.action_lim_low:
            motor_act[3] = self.action_lim_low
        """




        if self.motor_model_activated:
            motor_ref_l = np.sqrt((motor_act[0] + 1) / 2)
            motor_ref_r = np.sqrt((motor_act[1] + 1) / 2)
            #print(motor_ref_l," ", motor_ref_r)
            self.motor_speed_l = (4 * self.time_step / self.motor_settling_time) * (motor_ref_l - self.motor_speed_l) + self.motor_speed_l
            self.motor_speed_r = (4 * self.time_step / self.motor_settling_time) * (motor_ref_r - self.motor_speed_r) + self.motor_speed_r
            f_max = 0.5 * 9.81 * self.total_mass * self.thrust_to_weight_ratio
            thrust_l = f_max * self.motor_speed_l * self.motor_speed_l
            thrust_r = f_max * self.motor_speed_r * self.motor_speed_r
            unit_vecy = body.GetWorldVector(b2.b2Vec2(0, 1))  # Apply forces
            body.ApplyForce(force=thrust_l * unit_vecy, point=(body.GetWorldPoint(b2.b2Vec2(-self.witdhPlatform, 0))),
                            wake=True)  # left rotor
            body.ApplyForce(force=thrust_r * unit_vecy, point=(body.GetWorldPoint(b2.b2Vec2(self.witdhPlatform, 0))),
                            wake=True)  # right rotor
            #self.world.Step(self.time_step, self.vel_iters, self.pos_iters)

        else:
            unit_vecy = body.GetWorldVector(b2.b2Vec2(0, 1)) #get world vector of the body y axis 
            #print( motor_act[0],"  ", motor_act[1])
            forceLeft=(self.f_0 /2  + motor_act[0]*self.actionToForceConverter)
            forceRight=(self.f_0 /2  + motor_act[1]*self.actionToForceConverter)


            #forceLeft=(self.f_0 /2  )
            #forceRight=(self.f_0 /2  )
            #print(forceLeft,"  ",forceRight," ",self.totalForce/2.0)
            forceLeft=self.limitAppliedForce(forceLeft)
            forceRight=self.limitAppliedForce(forceRight)
            forceLeft=forceLeft*unit_vecy
            forceRight=forceRight*unit_vecy
            #print(forceLeft," ", forceRight)
            jointElbowleftFingerleft.motorSpeed= 4*motor_act[2]
            jointElbowrightFingerright.motorSpeed= -1*4*motor_act[2]

            jointtailbody.motorSpeed = motor_act[3]*self.tailLimitAngle*2
            #jointtailbody.motorSpeed = motor_act[3]  #this is used for pid 


            #print(motor_act[3])

            """if (self.countx == 0):
                jointElbowleftFingerleft.motorSpeed=  4
                jointElbowrightFingerright.motorSpeed= -4
                #self.countx = 1
            else:
                jointElbowleftFingerleft.motorSpeed=  -4
                jointElbowrightFingerright.motorSpeed= 4
                self.countx = 0"""

            """
            body.ApplyForce(force = (self.f_0/2.0)*unit_vecy ,point=(body.GetWorldPoint(b2.b2Vec2(-self.witdhPlatform,0))),
                            wake=True) #left motor
            body.ApplyForce(force = (self.f_0/2.0) *unit_vecy ,point=(body.GetWorldPoint(b2.b2Vec2(self.witdhPlatform,0))),
                            wake=True) #right motor
            jointElbowleftFingerleft.motorSpeed= 4*1
            jointElbowrightFingerright.motorSpeed= -1*4*1

            jointtailbody.motorSpeed = 4*-1
            """
            
            
            body.ApplyForce(force = forceLeft ,point=(body.GetWorldPoint(b2.b2Vec2(-self.witdhPlatform,0))),
                            wake=True) #left motor
            body.ApplyForce(force = forceRight ,point=(body.GetWorldPoint(b2.b2Vec2(self.witdhPlatform,0))),
                            wake=True) #right motor

            self.world.Step(self.time_step, self.vel_iters, self.pos_iters)


        currentPos= np.array([body.position[0], body.position[1]])
        if (self.ContactListener.IsGripped):
            gripmode = 1
            #print(gripmode)
        else:
            gripmode = 0


        vel = body.linearVelocity  # new state
        pos =  currentPos #- self.relativePos #-self.desPos
        posBox =  bodybox.position #-self.relativePos #b2.b2Vec2(0,-0.25) #bodybox.position
        ang = body.angle
        ang=self.limitAngleValue(ang)
        angleBox = bodybox.angle
        angleBox = self.limitAngleValue(angleBox)

        #print(jointtailbody.speed)
        #print(np.degrees(ang)," ",np.degrees(body.angle))
        avel = body.angularVelocity
        leftJointAngle= jointElbowleftFingerleft.angle
        rightJointAngle = jointElbowrightFingerright.angle
        tailJointAngle = jointtailbody.angle
        avelTail = jointtailbody.speed
        posFingerEndPoint = fingerleft.GetWorldPoint(b2.b2Vec2(self.witdhFinger, -self.heightFinger)) #-self.relativePos
        posBoxFingerLeftPoint = bodybox.GetWorldPoint(b2.b2Vec2(-self.witdhBox, 0))
        #print (jointtailbody.motorSpeed," val",jointtailbody.speed, " " ,motor_act[3]," " ,tailJointAngle, " " ,ang)
        #print(leftJointAngle," ",rightJointAngle )
        #print(self.stepnum)

        #print(forceLeft," ",forceRight)
        self.prev_st = self.st
        
        if (self.IsRandomDesPosTrain ):
            self.st = np.append([pos,posFingerEndPoint,posBox,self.desBoxPosObs, vel], [ang, avel,angleBox,gripmode,leftJointAngle,tailJointAngle]).reshape(16)
        else:
            self.st = np.append([pos,posFingerEndPoint,posBox, vel], [ang, avel,angleBox,gripmode,leftJointAngle,tailJointAngle]).reshape(14)

        self.out_of_borders()
        self.out_of_bordersBox()
        self.crash_Occured()

        hardStopValue=0
        if self.hard_stop_enable:  # episode finishing conditions
            #if  posBox[1] > self.desBoxPos[1]: #use this for above sth movement
            if (np.abs(posBox[0]-self.desBoxPos[0]) < 0.1 and np.abs(posBox[1]-self.desBoxPos[1]) < 0.1):
                #hardStopValue=1
                self.hard_stop_episode = True
        self.stepnum += 1
        rw = self.reward_calc()
        #print(rw)


        self.prevAct=self.act


        if self.stepnum > self.step_per_episode or self.OutOfBorder or self.hard_stop_episode or self.CrashOccured or self.OutOfBorderBox:
            #print("false stop",self.stepnum , " " ,self.CrashOccured," " , self.OutOfBorderBox)
            if (self.stepnum > self.step_per_episode) :
                hardStopValue = 2
            if (self.OutOfBorder):
                hardStopValue =3
            if (self.hard_stop_episode):
                hardStopValue =1
            if (self.CrashOccured):
                hardStopValue =4
            if (self.OutOfBorderBox):
                hardStopValue = 5
            self.dn = True
        else:
            self.dn = False

        self.desPos = np.array([bodybox.position[0],bodybox.position[1]-self.boxDistanceToOrigin])  # destination
        self.desPosFinger = np.array([posBoxFingerLeftPoint[0],posBoxFingerLeftPoint[1]])  # destination finger
        return self.st, rw, self.dn, {'finstate': hardStopValue}

    def reset(self):
        body = self.world.bodies[0]
        tail = self.world.bodies[1]
        bodybox = self.world.bodies[2]
        elbowleft= self.world.bodies[4]
        elbowright= self.world.bodies[5]
        fingerleft=self.world.bodies[6]
        fingerright=self.world.bodies[7]

        jointtailbody = self.world.joints[0]
        jointTailElbowleft = self.world.joints[1]
        jointTailElbowright =self.world.joints[2]
        jointElbowleftFingerleft =self.world.joints[3]
        jointElbowrightFingerright =self.world.joints[4]

        self.checkPriorKnowledgeStart()


        self.posInitPlatform= b2.b2Vec2( np.random.uniform(low=self.init_dist_pos_low_x,
                                                                    high=self.init_dist_pos_high_x),
                                   np.random.uniform(low=self.init_dist_pos_low_y,
                                                                    high=self.init_dist_pos_high_y))


        
        if self.set_init_pos is not None:
            self.posInitPlatform = b2.b2Vec2(self.set_init_pos[0],self.set_init_pos[1])

        body.position = self.posInitPlatform
        body.linearVelocity = b2.b2Vec2(np.random.uniform(low=-1, high=1),
                                        np.random.uniform(low=-1, high=1)) * self.init_dist_vel
        body.angle = np.random.uniform(low=-1, high=1) * self.init_dist_ang
        body.angularVelocity = np.random.uniform(low=-1, high=1) * self.init_dist_avel
        
        self.setPositions(self.posInitPlatform)

        elbowright.position = self.posInitElbowRight
        elbowright.linearVelocity = b2.b2Vec2(0, 0)
        elbowright.angularVelocity = 0
        elbowright.angle = 0

        
        elbowleft.position=self.posInitElbowLeft
        elbowleft.linearVelocity = b2.b2Vec2(0, 0)
        elbowleft.angularVelocity = 0
        elbowleft.angle = 0

       
        jointTailElbowleft.motorSpeed=0

        bodybox.position = self.posInitBox
        bodybox.linearVelocity = b2.b2Vec2(0, 0)
        bodybox.angularVelocity = 0
        bodybox.angle = 0

        
        jointTailElbowright.motorSpeed=0

        fingerleft.position = self.posInitFingerLeft
        fingerleft.linearVelocity = b2.b2Vec2(0, 0)
        fingerleft.angularVelocity = 0
        fingerleft.angle = self.fingerStartAngle

        
        jointElbowleftFingerleft.motorSpeed=0

        fingerright.position =self.posInitFingerRight
        fingerright.linearVelocity = b2.b2Vec2(0, 0)
        fingerright.angularVelocity = 0
        fingerright.angle = -self.fingerStartAngle

        jointElbowrightFingerright.motorSpeed=0

        #print(tail.position)
        tail.position = self.posInitTail
        tail.linearVelocity = b2.b2Vec2(0, 0)
        tail.angle = self.tailStartAngle
        tail.angularVelocity = 0

        jointtailbody.motorSpeed=0



        vel = body.linearVelocity
        pos = body.position 
        posBox =  bodybox.position #b2.b2Vec2(0,-0.25) #bodybox.position
        angleBox = bodybox.angle
        ang = body.angle
        avel = body.angularVelocity
        leftJointAngle= jointElbowleftFingerleft.angle
        rightJointAngle = jointElbowrightFingerright.angle
        avelTail= jointtailbody.speed
        tailJointAngle =jointtailbody.angle



        posFingerEndPoint = fingerleft.GetWorldPoint(b2.b2Vec2(self.witdhFinger, -self.heightFinger))
        posBoxFingerLeftPoint = bodybox.GetWorldPoint(b2.b2Vec2(-self.witdhBox, 0))
        #print (posFingerEndPoint)

        if self.IsRandomDesPosTrain:
            #self.desBoxPosObs = b2.b2Vec2( np.random.uniform(low=self.desPosBoxLow_x,
            #                                                        high=self.desPosBoxHigh_x), 0.1)   #for comparasion small region
            self.desBoxPosObs = b2.b2Vec2( np.random.uniform(low=self.desPosBoxLow_x,
                                                                    high=self.desPosBoxHigh_x),
                                            np.random.uniform(low = self.desPosBoxLow_y,
                                                                    high=self.desPosBoxHigh_y))
            if self.set_des_pos is not None:
                self.desBoxPosObs = b2.b2Vec2(self.set_des_pos[0],self.set_des_pos[1])
            self.desBoxPos = np.array([posBox[0]+ self.desBoxPosObs[0],posBox[1]-self.boxDistanceToOrigin + self.desBoxPosObs[1]])
            self.desBoxPosObs = b2.b2Vec2( self.desBoxPos[0], self.desBoxPos[1] )


        self.desPos = np.array([posBox[0],posBox[1]-self.boxDistanceToOrigin])  # destination
        self.desPosFinger = np.array([posBoxFingerLeftPoint[0],posBoxFingerLeftPoint[1]])  # destination finger

        gripmode= 0;

        if (self.IsRandomDesPosTrain ):
            self.st = np.append([pos,posFingerEndPoint,posBox,self.desBoxPosObs, vel], [ang, avel,angleBox,gripmode,leftJointAngle,tailJointAngle]).reshape(16)
        else:
            self.st = np.append([pos,posFingerEndPoint,posBox, vel], [ang, avel,angleBox,gripmode,leftJointAngle,tailJointAngle]).reshape(14)
        
        
        self.prev_st = self.st
        self.stepnum = 0
        self.dn = False
        self.hard_stop_episode = False
        self.OutOfBorder=False
        self.OutOfBorderBox=False
        self.CrashOccured = False 
        self.ContactListener.ResetValues()
        self.fg = None
        self.ax = None
        return self.st
 

    def render(self, mode='human', close=False):
	#draw positions of vertices with filling the below array (polygon fill function)
        arrayx = [0,0,0,0,0]
        arrayy = [0,0,0,0,0]

    #get bodies from world
        body = self.world.bodies[0]
        tail = self.world.bodies[1]
        boxs  = self.world.bodies[2]
        ground = self.world.bodies[3]
        elbowleft = self.world.bodies[4]
        elbowright = self.world.bodies[5]
        fingerleft = self.world.bodies[6]
        fingerright = self.world.bodies[7]

        jointtailbody = self.world.joints[0]
        jointTailElbowleft = self.world.joints[1]
        jointTailElbowright =self.world.joints[2]
        jointElbowleftFingerleft =self.world.joints[3]
        jointElbowrightFingerright =self.world.joints[4]

        if self.fg == None:
            self.fg, self.ax = plt.subplots(1,1)
            print(self.fg)
        self.ax.cla()
        self.ax.plot([-5, -5, 5, 5, -5], [-5, 5, 5, -5, -5])

        bodyfixtures=body.massData
        tailfixtures=tail.massData
        boxsfixtures=boxs.massData
        tailpos=tail.position
        #print("JELFL: ",jointElbowleftFingerleft.speed, " JERFR: ",jointElbowrightFingerright.speed)
        #print("TB ",jointtailbody.angle," TEL ",jointTailElbowleft.angle," TER ",jointTailElbowright.angle," ELFL ", jointElbowleftFingerleft.angle, " ERFR ",jointElbowrightFingerright.angle)
        #print(tailpos)
        #print('body',bodyfixtures)
        #print('tail',tailfixtures)
        #print('box',boxsfixtures)
        #self.ax.plot(self.desPos[0], self.desPos[1], marker="x") #destination

        self.ax.plot(self.desBoxPos[0],self.desBoxPos[1],marker="o")

        self.ax.plot(self.desPosFinger[0],self.desPosFinger[1],marker="x")

        arrayx,arrayy=self.draw_polygon(tail,self.witdhTail,self.heightTail) #tail
        self.ax.fill(arrayx,arrayy, color="brown")

        arrayx,arrayy=self.draw_polygon(body,self.witdhPlatform,self.heightPlatfrom) #platform
        self.ax.fill(arrayx,arrayy, color="brown")
        
        arrayx,arrayy=self.draw_polygon(boxs,self.witdhBox,self.heightBox) #box1
        self.ax.fill(arrayx,arrayy)

        arrayx,arrayy=self.draw_polygon(ground,self.witdhGround1,self.heightGround1+0.03) #ground1
        self.ax.fill(arrayx,arrayy)
    
        arrayx,arrayy=self.draw_polygon(elbowleft,self.witdhElbow,self.heightElbow) #elbowleft
        self.ax.fill(arrayx,arrayy,color="brown")

        arrayx,arrayy=self.draw_polygon(elbowright,self.witdhElbow,self.heightElbow) #elbowright
        self.ax.fill(arrayx,arrayy,color="brown")

        arrayx,arrayy=self.draw_polygon(fingerleft,self.witdhFinger,self.heightFinger) #fingerleft
        self.ax.fill(arrayx,arrayy,color="brown")

        arrayx,arrayy=self.draw_polygon(fingerright,self.witdhFinger,self.heightFinger) #fingerright
        self.ax.fill(arrayx,arrayy,color="brown")



        pos = body.position
        unit_vecx = body.GetWorldVector(b2.b2Vec2(self.witdhPlatform, 0))
        unit_vecy = body.GetWorldVector(b2.b2Vec2(0, self.witdhPlatform))
        pos1 = pos - unit_vecx   # left rotor
        pos2 = pos + unit_vecx   # right rotor

        if self.motor_model_activated:
            motor_ref_l = np.sqrt((self.act[0] + 1) / 2)
            motor_ref_r = np.sqrt((self.act[1] + 1) / 2)
            self.motor_speed_l = (4 * self.time_step / self.motor_settling_time) * (motor_ref_l - self.motor_speed_l) + self.motor_speed_l
            self.motor_speed_r = (4 * self.time_step / self.motor_settling_time) * (motor_ref_r - self.motor_speed_r) + self.motor_speed_r
            f_max = 0.5 * 9.81 * self.total_mass * self.thrust_to_weight_ratio
            thrust_l = f_max * self.motor_speed_l * self.motor_speed_l
            thrust_r = f_max * self.motor_speed_r * self.motor_speed_r
            unit_vecy = body.GetWorldVector(b2.b2Vec2(0, 1))  # Apply forces
            self.ax.plot([pos1[0], pos1[0] + 0.1 * thrust_l * unit_vecy[0]],
                     [pos1[1], pos1[1] + 0.1 * thrust_l * unit_vecy[1]],
                     color="red")  # left rotor
            self.ax.plot([pos2[0], pos2[0] + 0.1 * thrust_r * unit_vecy[0]],
                     [pos2[1], pos2[1] + 0.1 * thrust_r * unit_vecy[1]],
                     color="red")  # right rotor

        else:
            f_max = self.total_mass*self.gravitational_acc*self.thrust_to_weight_ratio
            act0 = (self.act[0]+1)*f_max/4
            act1 = (self.act[1]+1)*f_max/4


            self.ax.plot([pos1[0], pos1[0] + 0.1 * act0 * unit_vecy[0]],
                     [pos1[1], pos1[1] + 0.1 * act0 * unit_vecy[1]],
                     color="black")  # left rotor
            self.ax.plot([pos2[0], pos2[0] + 0.1 * act1 * unit_vecy[0]],
                     [pos2[1], pos2[1] + 0.1 * act1 * unit_vecy[1]],
                     color="black")  # right rotor
        # plt.plot(tail.position[0], tail.position[1], marker="x")

        # TODO show applied torque on image

        self.ax.axis([-self.axisValue, self.axisValue, -self.axisValue, self.axisValue])
        plt.pause(self.time_step / 100)
        self.ax.set_aspect('equal','box')
        self.ax.grid(b=True)
        #plt.savefig("/home/halil/Documents/halil/tez_plots/render_p2p_tail_cls/fig" + str(self.stepnum) + ".png")
        #plt.draw()

    def reward_calc(self):
        action_punishment = 0
        angle_punishment = 0
        body = self.world.bodies[0]
        box = self.world.bodies[2]
        fingerleft = self.world.bodies[6]
        
        grippedReward = 0;
        crashPunishment = 0;
        armPushPunishment = 0;
        outOfBorderPunishment = 0;
        #print(self.ContactListener.IsGripped," ",self.ContactListener.IsCrash)
        if (self.ContactListener.IsGripped):
            grippedReward =2.5;
        if(self.ContactListener.IsArmPush):
            armPushPunishment = 2.5;
        if (self.ContactListener.IsCrash):
            crashPunishment = (self.step_per_episode-self.stepnum)*self.ContactListener.IsCrash*0.75 #2.5
            """if (crashPunishment < 0):
                print (crashPunishment," ",self.ContactListener.IsCrash, " ",self.step_per_episode-self.stepnum)"""
            #crashPunishment = self.ContactListener.IsCrash*5

        if (self.OutOfBorder or self.OutOfBorderBox):
            outOfBorderPunishment = (self.step_per_episode-self.stepnum)*2.5

        if(np.abs(body.angle)>0.5):
            angle_punishment= -20

        if self.act[0] > 1.1 or self.act[1] > 1.1 or self.act[0] < -1.1 or self.act[1] < -1.1:
            action_punishment = -20
        hovering_actions = np.array([0,0,0,0])
        action_punishment -= 0.5 * la.norm(self.act - hovering_actions, ord=1)  # *(self.stepnum**2 / 10000)
        #action_punishment=la.norm(self.act - self.prevAct, ord=1)
        # print("act pun: ",action_punishment,self.act, hovering_actions)
        posBox = np.array([self.st[4],self.st[5]])
        #print(posbox)
        pos = np.array([self.st[0], self.st[1]])
        posFinger = np.array([self.st[2], self.st[3]])

        if (self.IsRandomDesPosTrain ):
            vel = np.array([self.st[8], self.st[9]])
            ang = self.st[10]
            avel = self.st[11]
        else:
            vel = np.array([self.st[6], self.st[7]])
            ang = self.st[8]
            avel = self.st[9]            
        

        ep = self.desPos - pos
        ev = self.desVel - vel
        epFinger = self.desPosFinger - posFinger
        epBox = self.desBoxPos-posBox
        #print(ep)
        


        if (self.ContactListener.IsGripped):
            epBox = epBox  #grip state i halen istenen bir state ama ep - oldugu icin biraz daha dusuk eklenti
            ep= epFinger #np.array([0, 0])
            
        else:
            epBox=epBox #np.array([0, 0]);
            ep=epFinger


        
        extra_pose_reward = 0
        test=la.norm(ep)
        test2= la.norm(ep,ord=1)
        test3 = la.norm(ang)

        if self.hard_stop_episode:
            return  (self.step_per_episode-self.stepnum)*2.5*2
        else:
            return  -(1.5 * la.norm(ep, ord=1) + 0.2 * la.norm(ev, ord=1) + 0.5*la.norm(ang) ) + 0.05*action_punishment + grippedReward -crashPunishment - armPushPunishment - outOfBorderPunishment -2*la.norm(epBox,ord=1)
        """if self.OutOfBorder:
            print(self.stepnum)
            return -(2.0 * la.norm(ep, ord=1) + 0.2 * la.norm(ev, ord=1) + 0.5*la.norm(ang)) + 0.05*action_punishment -(self.step_per_episode-self.stepnum)*0.1"""
        
        #print(ep," ",test," ",test2," ",ang," ",test3)
        # if la.norm(ep,ord=1) < 0.2:
        #   extra_pose_reward = 25
        #if self.hard_stop_episode:
        #    return  2.5- (2.0 * la.norm(ep, ord=1) + 0.2 * la.norm(ev, ord=1) + 0.1*la.norm(ang)) + 0.1*action_punishment 
        #return  -(2 * la.norm(ep, ord=1) + 0.2 * la.norm(ev, ord=1) + 0.5*la.norm(ang) + 5*la.norm(epBox,ord=1)) + 0.05*action_punishment + grippedReward -crashPunishment - armPushPunishment
        #return  -(2 * la.norm(ep, ord=1) + 0.2 * la.norm(ev, ord=1) + 0.5*la.norm(ang) ) + 0.05*action_punishment + grippedReward -crashPunishment - armPushPunishment

    def out_of_borders(self):
        posx = self.st[0]
        posy = self.st[1]
        
        

        if np.abs(posx) > self.outOfBorderPos or np.abs(posy) > self.outOfBorderPos:
            self.OutOfBorder=True

        else:
            self.OutOfBorder=False


    def out_of_bordersBox(self):
        box = self.world.bodies[2]
        posBoxx = box.position[0]
        posBoxy = box.position[1]



        if np.abs(posBoxx) > self.outOfBorderPos or np.abs(posBoxy) > self.outOfBorderPosBox:
            self.OutOfBorderBox=True
        else:
            self.OutOfBorderBox=False


    def crash_Occured(self):
        if (self.ContactListener.IsCrash >0 ):
            self.CrashOccured = True
        else:
            self.CrashOccured = False 


    #box 2d only moves the center of mas and the angle to draw one need to get the vertices using getworld
    def draw_polygon(self,PolyBody,PolyWidth,PolyHeight):
        pointsX = [0,0,0,0,0]
        pointsY = [0,0,0,0,0]

        centerpos=PolyBody.position

        #box 2d only moves the center of mas and the angle to draw one need to get the vertices using getworld
        BoxPoint0=PolyBody.GetWorldVector(b2.b2Vec2(-PolyWidth, -PolyHeight))
        BoxPoint1=PolyBody.GetWorldVector(b2.b2Vec2(PolyWidth, -PolyHeight))
        BoxPoint2=PolyBody.GetWorldVector(b2.b2Vec2(PolyWidth, PolyHeight))
        BoxPoint3=PolyBody.GetWorldVector(b2.b2Vec2(-PolyWidth, PolyHeight))

        pointsX[0]= centerpos[0]+ BoxPoint0[0]
        pointsX[1]= centerpos[0]+ BoxPoint1[0]
        pointsX[2]= centerpos[0]+ BoxPoint2[0]
        pointsX[3]= centerpos[0]+ BoxPoint3[0]
        pointsX[4]= pointsX[0]

        pointsY[0]= centerpos[1]+ BoxPoint0[1]
        pointsY[1]= centerpos[1]+ BoxPoint1[1]
        pointsY[2]= centerpos[1]+ BoxPoint2[1]
        pointsY[3]= centerpos[1]+ BoxPoint3[1]
        pointsY[4]= pointsY[0]

        return pointsX,pointsY
    #when the body of platform is changed directly all the other shapes must also move with it which is why this function is written
    def setPositions(self,PlatformStartingPos):
        self.posInitTail = PlatformStartingPos + b2.b2Vec2(0,-self.heightTail)
        self.posInitElbowLeft = self.posInitTail + b2.b2Vec2(-self.witdhElbow-self.witdhTail,-self.heightTail)
        self.posInitElbowRight = self.posInitTail + b2.b2Vec2(+self.witdhElbow+self.witdhTail,-self.heightTail)
        self.posInitFingerLeft = self.posInitElbowLeft + b2.b2Vec2(-self.witdhElbow,-self.heightFinger)
        self.posInitFingerRight = self.posInitElbowRight + b2.b2Vec2(self.witdhElbow,-self.heightFinger)
        if (self.IsStartWithPrior == False):
            self.posInitBox = b2.b2Vec2(0,self.boxDistanceToOrigin)
        else:
            self.posInitBox = PlatformStartingPos + b2.b2Vec2(0,self.boxDistanceToOrigin)
        #self.posInitBox = b2.b2Vec2(0,self.boxDistanceToOrigin)
        #self.posInitBox = PlatformStartingPos + b2.b2Vec2(0,self.boxDistanceToOrigin)


    def limitAppliedForce(self,AppliedForce):
        #print(AppliedForce)
        resultForce=AppliedForce
        if (resultForce > self.totalForce/2.0):
            resultForce= self.totalForce/2.0
        elif (resultForce < 0):
            resultForce=0
        return resultForce
    def limitAngleValue(self,Angle):
        resultAngle=Angle
        if (resultAngle > np.pi):
            resultAngle=(resultAngle % (2*np.pi))
            if (resultAngle > np.pi):
                resultAngle=resultAngle-2*np.pi
        elif (resultAngle < -np.pi):
            resultAngle=(resultAngle % (-2*np.pi))
            if (resultAngle < -np.pi):
                resultAngle=resultAngle+2*np.pi
        return resultAngle
    def setDesPos(self):
        #self.desPos = np.array([boxs.position[0], boxs.position[1]-self.boxDistanceToOrigin])  # destination
        self.desPos = np.array([0,0])  # destination

    def checkPriorKnowledgeStart(self):
        self.IsStartWithPrior = False
        self.probPrior = np.random.uniform(low=0,high=1)
        if (self.probPrior < self.priorKnowledgeChance):
            self.IsStartWithPrior = True #for box start position
            self.fingerStartAngle = self.fingerLimitAngle #for finger start angle

        else:
            self.IsStartWithPrior = False 
            self.fingerStartAngle = 0.0
        #print(self.IsStartWithPrior, " " ,self.probPrior, " " ,self.priorKnowledgeChance)

# TODO add environment only for attitude control
