from gym.core import Env
from gym.spaces import Discrete, Tuple
from gym.spaces.space import Space

import numpy as np


WIDTH = 4
HEIGHT = 3

POS_GOAL = (3, 2)
NEG_GOAL = (3, 1)
BARRIER = (1, 1)


class GridSpace(Space):

    def __init__(self):
        self.nested = Tuple((Discrete(WIDTH), Discrete(HEIGHT)))

    def contains(self, x):
        return self.nested.contains(x) and x != BARRIER

    def sample(self, x):
        ans = BARRIER
        while ans == BARRIER:
            ans = self.nested.sample()
        return ans


class Grid4x3(Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = GridSpace()
        self.reward_range = [-1, 1]
        self.pos = (0, 0)

    def close(self):
        pass

    def render(self, mode='human'):
        mat = [[' ' for _ in range(WIDTH+2)] for _ in range(HEIGHT+2)]

        # Draw screen borders.
        for i in range(1, HEIGHT+1):
            mat[i][0] = mat[i][WIDTH+1] = '█'
        for j in range(1, WIDTH+1):
            mat[0][j] = mat[HEIGHT+1][j] = '█'
        for i in {0, HEIGHT+1}:
            for j in {0, WIDTH+1}:
                mat[i][j] = '█'

        # The agent's position.
        (j, i) = self.pos; mat[i+1][j+1] = 'o'

        # Goals.
        (j, i) = POS_GOAL; mat[i+1][j+1] = '+'
        (j, i) = NEG_GOAL; mat[i+1][j+1] = '-'

        # Barrier.
        (j, i) = BARRIER; mat[i+1][j+1] = '█'

        # Print buffer to the screen.
        for row in reversed(mat):
            for x in row:
                print(x, end='')
            print()


    def reset(self):
        self.pos = (0, 0)
        return self.observation_space.sample()

    def step(self, action):
        rand = np.random.random()
        if rand > 0.9:
            action += 1
        elif rand > 0.8:
            action -= 1
        action %= 4

        (j, i) = self.pos
        i += (action == 0 and i < HEIGHT-1) - (action == 2 and i > 1)
        j += (action == 1 and j < WIDTH-1)  - (action == 3 and j > 1)
        if (j, i) != BARRIER:
            self.pos = (j, i)

        if self.pos == NEG_GOAL:
            return (self.pos, -1, True, dict())
        elif self.pos == POS_GOAL:
            return (self.pos, +1, True, dict())
        else:
            return (self.pos, -0.04, False, dict())
