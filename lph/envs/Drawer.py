from lph.envs.AbsEnv import GraphEnv


class DrawerEnv(GraphEnv):
    """
    Drawer tidying environment.

    Actions are:
                    0. Open drawer
                    1. Remove spoon from cup
                    2. Push box to the side
                    3. Place pen in drawer
                    4. Place cup in drawer
                    5. Close drawer

    State dims are the same as actions.
    """
    conditions = [[], [], [0], [0],  # ............. 0, 1, 2, 3
                  (1, 2), (3, 4)]  # ............ 4, 5

    # A few useful goals
    goal_pen_in = ([3], [1])
    goal_box_out = ([2], [1])
    goal_cup_in = ([4], [1])
    goal_close = ([5], [1])

    # Suggested curriculum
    curriculum = [
        goal_pen_in,
        goal_close
    ]

    def __init__(self, stochastic_reset=False, goal=None):
        super().__init__(DrawerEnv.conditions,
                         DrawerEnv.curriculum,
                         stochastic_reset, goal=goal)
