from gym.envs.registration import register

register(
    id='quad-withArm_1Dof-v0',
    entry_point='AerialManipulatorDRL.envs:QuadWithArm1Dof',

)
