from enum import Enum
import enum
import numpy as np
import random

ActionResult = (np.ndarray, float, bool, enum, enum)

class TicTacToe(object):
    def __init__(self, length=3) -> None:
        self.row = length
        self.col = length
        self.field = np.array([SquareState.NOTHING for _ in range(self.row*self.col)])
        self.step_count = 0
        self.action_num = length * length

    def step(self, action) -> ActionResult:
        '''
        input: これから打つ升目
        return: フィールドの状況, 報酬, 終了フラグ, どっちのターンだったか, 次のターンプレーヤー
        '''
        # CROSS is first
        if (self.step_count % 2 == 0):
            square_state = SquareState.CROSS
            turn_player = Agents.CROSS
            next_player = Agents.CIRCLE
        else:
            square_state = SquareState.CIRCLE
            turn_player = Agents.CIRCLE
            next_player = Agents.CROSS

        # もうすでに打たれているマス目を選択した場合は負け
        disavailable = self.get_disavailable_select_actions()
        if (action in disavailable):
            return (TicTacToe._enum_to_number(self.field), -1.0, True, True, turn_player, turn_player)

        self.field[action] = square_state

        result_status = self._check(square_state)

        done = False
        reward = 0.0
        if result_status == ResultStatus.DRAW:
            reward = 0.0
            done = True
        elif result_status == ResultStatus.WIN:
            reward = 1.0
            done = True
        else:
            reward = -1.0
            done = False

        self.step_count += 1
        return (TicTacToe._enum_to_number(self.field), reward, done, False, turn_player, next_player)

    def reset(self) -> np.ndarray:
        self.field = np.array([SquareState.NOTHING for _ in range(self.row*self.col)])
        self.step_count = 0
        return TicTacToe._enum_to_number(self.field)

    def get_available_select_action(self) -> list:
        return np.where(self.field == SquareState.NOTHING)[0]

    def get_disavailable_select_actions(self) -> list:
        return np.where(self.field != SquareState.NOTHING)[0]

    def get_field(self):
        return TicTacToe._enum_to_number(self.field)

    def easy_display(self) -> None:
        for r in range(self.row):
            for c in range(self.col):
                print("{}".format(self.field[r * self.col + c].value), end="")
                if (c == self.col - 1):
                    print("")
                else:
                    print(", ", end="")

    def _check(self, square_state) -> 'ResultStatus':
        matrix = []
        for r in range(self.row):
            lst = []
            for c in range(self.col):
                lst.append(self.field[r * self.col + c])
            matrix.append(lst)
        matrix = np.array(matrix)

        count = 0

        # horizontal check
        for r in range(self.row):
            count = np.count_nonzero(matrix[r] == square_state)
            if (count == self.row):
                return ResultStatus.WIN

        # vertical check
        for c in range(self.col):
            count = np.count_nonzero(matrix.T[c] == square_state)
            if (count == self.col):
                return ResultStatus.WIN

        # diagonal check
        sampling_diagonal = self._sampling_diagonal()
        for n in range(len(sampling_diagonal)):
            count = np.count_nonzero(sampling_diagonal[n] == square_state)
            if (count == self.row):
                return ResultStatus.WIN

        if np.count_nonzero(self.field == SquareState.NOTHING) == 0:
            return ResultStatus.DRAW

        return ResultStatus.PROGRESS

    def _sampling_diagonal(self) -> np.ndarray:
        lst1 = []
        lst2 = []
        for n in range(self.row):
            lst1.append(self.field[n * self.col + n])
            lst2.append(self.field[n * self.col + (self.col - 1 - n)])
        return np.array([lst1, lst2])

    def _enum_to_number(ary) -> np.ndarray:
        return np.array([x.value for x in ary])

class SquareState(Enum):
    NOTHING = 0
    CROSS = 1
    CIRCLE = 2

class ResultStatus(Enum):
    '''
    勝ちの判定のみあれば逆サイドは負けなので判定可能
    '''
    PROGRESS = 0
    DRAW = 1
    WIN = 2

class Agents(Enum):
    CROSS = "cross"
    CIRCLE = "circle"

def main():
    env = TicTacToe()
    env.reset()
    
    print("reset env")
    env.easy_display()
    print("")

    env.reset()
    done = False
    step_count = 0
    while not done:
        step_count += 1
        action_num = len(env.get_available_select_action())
        chose_action = random.randrange(action_num)
        target_index = env.get_available_select_action()[chose_action]
        _, reward, done, kind = env.step(target_index)
        print("{} step".format(step_count))
        env.easy_display()
        print("")

    if (reward == 1.0):
        print("draw")
    else:
        print("{} win!".format(kind.value))

if __name__ == '__main__':
    main()
