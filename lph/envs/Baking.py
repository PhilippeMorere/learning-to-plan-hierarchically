from lph.envs.AbsEnv import GraphEnv


class BakingEnv(GraphEnv):
    """
    Baking environment, making chocolate or sultana cookies

    Actions are:
                    0. Get egg
                    1. Break egg
                    2. Whisk egg
                    3. Pour egg
                    4. Get flour
                    5. Add flour
                    6. Get sugar
                    7. Pour sugar
                    8. Get salt
                    9. Add salt
                    10. Get oil
                    11. Pour oil
                    12. Get milk
                    13. Pour milk
                    14. Mix liquids
                    15. Mix solids
                    16. Mix batter
                    17. Get butter
                    18. Get tray
                    19. Oil tray
                    20. Get chocolate
                    21. Make chocolate chips
                    22. Mix in chocolate chips
                    23. Pour chocolate batter
                    24. Turn oven on
                    25. Cook chocolate cookies
                    26. Get sultana
                    27. Mix in sultana
                    28. Poor sultana batter
                    29. Cook sultana cookies

    State dims are the same as actions.
    """
    conditions = [[], [0], [1], [2],  # ............. 0, 1, 2, 3
                  [], [4], [], [6],  # .............. 4, 5, 6, 7
                  [], [8], [], [10],  # ............. 8, 9, 10, 11
                  [], [12],  # ...................... 12, 13
                  (3, 11, 13), (5, 7, 9),  # ........ 14, 15
                  (14, 15), [], [], (17, 18),  # .... 16, 17, 18, 19
                  [], [20], (16, 21),  # ............ 20, 21, 22
                  (19, 22), [], (23, 24),  # ........ 23, 24, 25
                  [], (16, 26), (19, 27), [28]]  # .. 26, 27, 28, 29

    # A few useful goals
    goal_break_egg = ([1], [1])
    goal_pour_egg = ([3], [1])
    goal_pour_four = ([5], [1])
    goal_pour_sugar = ([7], [1])
    goal_add_salt = ([9], [1])
    goal_pour_oil = ([11], [1])
    goal_pour_milk = ([13], [1])
    goal_mix_liquids = ([14], [1])
    goal_mix_solids = ([15], [1])
    goal_mix_batter = ([16], [1])
    goal_oil_tray = ([19], [1])
    goal_make_choc_chips = ([21], [1])
    goal_mix_choc_chips = ([22], [1])
    goal_pour_choc_batter = ([23], [1])
    goal_cook_choc_cookie = ([25], [1])
    goal_pour_sultana_batter = ([28], [1])
    goal_cook_sultana_cookie = ([29], [1])

    # Suggested curriculum
    curriculum = [
        goal_break_egg,
        goal_pour_egg,
        goal_pour_four,
        goal_pour_sugar,
        goal_add_salt,
        goal_pour_oil,
        goal_pour_milk,
        goal_mix_liquids,
        goal_mix_solids,
        goal_mix_batter,
        goal_oil_tray,
        goal_make_choc_chips,
        goal_mix_choc_chips,
        goal_pour_choc_batter,
        goal_cook_choc_cookie,
        goal_pour_sultana_batter,
        goal_cook_sultana_cookie
    ]

    def __init__(self, stochastic_reset=False, goal=None):
        super().__init__(BakingEnv.conditions, BakingEnv.curriculum,
                         stochastic_reset, goal=goal)
