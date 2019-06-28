from lph.envs.AbsEnv import GraphEnv
from lph.utils import SparseState


class MiningCraftingEnv(GraphEnv):
    """
    Mining and crafting Minecraft inspired domain.

    Actions are:
                    0. Get wood
                    1. Get stone
                    2. Get string
                    3. Make firewood
                    4. Make stick
                    5. Make arrow
                    6. Make bow
                    7. Make stone pickaxe
                    8. Get coal
                    9. Get iron
                    10. Get siver
                    11. Light furnace
                    12. Smelt iron
                    13. Smelt silver
                    14. Make iron pickaxe
                    15. Make silverware
                    16. Get gold
                    17. Get diamond
                    18. Smelt gold
                    19. Make earrings
                    20. Make goldware
                    21. Make necklace

    State dims:
                    0. Has wood
                    1. Has stone
                    2. Has string
                    3. Has firewood
                    4. Has stick
                    5. Has arrow
                    6. Has bow
                    7. Has stone pickaxe
                    8. Has coal
                    9. Has iron ore
                    10. Has sliver ore
                    11. Furnace lit
                    12. Has iron
                    13. Has silver
                    14. Has iron pickaxe
                    15. Has silverware
                    16. Has gold ore
                    17. Has diamond
                    18. Has gold
                    19. Has earrings
                    20. Has goldware
                    21. Has necklace
    """
    conditions = [[], [], [],  # ................ 0, 1, 2
                  [0], [0], (1, 2), (0, 2),  # .. 3, 4, 5, 6
                  (1, 4),  # .................... 7
                  [7], [7], [7],  # ............. 8, 9, 10
                  [8],  # ....................... 11
                  (9, 11), (10, 11),  # ......... 12, 13
                  (4, 12), [13],  # ............. 14, 15
                  [14], [14],  # ................ 16, 17
                  (11, 16), (13, 17),  # ........ 18, 19
                  [18], (17, 18)]  # ............ 20, 21

    # A few useful goals
    goal_wood = ([0], [1])
    goal_stick = ([4], [1])
    goal_stone_pick = ([7], [1])
    goal_coal = ([8], [1])
    goal_iron_ore = ([9], [1])
    goal_silver_ore = ([10], [1])
    goal_furnace = ([11], [1])
    goal_smelt_iron = ([12], [1])
    goal_iron_pick = ([14], [1])
    goal_gold_ore = ([16], [1])
    goal_diamond = ([17], [1])
    goal_gold = ([18], [1])
    goal_earrings = ([19], [1])
    goal_goldware = ([20], [1])
    goal_necklace = ([21], [1])

    # Suggested curriculum
    curriculum = [
        goal_stick,
        goal_stone_pick,
        goal_coal,
        goal_furnace,
        goal_smelt_iron,
        goal_iron_pick,
        goal_gold,
        goal_necklace]

    def __init__(self, stochastic_reset=False, goal=None):
        super().__init__(MiningCraftingEnv.conditions,
                         MiningCraftingEnv.curriculum,
                         stochastic_reset, goal=goal)


class MiningCraftingEnvGym(MiningCraftingEnv):
    def __init__(self, stochastic_reset=False):
        goal = MiningCraftingEnv.curriculum[-1]
        d_s = len(MiningCraftingEnv.conditions)
        s_goal = SparseState(goal[0], goal[1], d_s)
        super().__init__(stochastic_reset, s_goal)
