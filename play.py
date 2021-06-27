import pygame
from pygame.locals import *
import sys
from env import SquareState

SCREEN_SIZE = 800
PADDING_SIZE = 10
BACK_GROUND_COLOR = (0, 0, 0)
LINE_COLOR = (255, 255, 255)
LINE_WIDTH = 5

class PointInfo(object):
    def __init__(self, point, field_size) -> None:
        self.point = point
        x, y = point
        size = SCREEN_SIZE // (field_size * 2)
        self.x_range = range(x - size, x + size)
        self.y_range = range(y - size, y + size)
        self.state = SquareState.NOTHING

    def isSetting(self) -> bool:
        return self.state == SquareState.NOTHING

def main():
    field_size = 3

    # フィールド情報の初期化
    field = []
    y = 0
    for row in range(field_size):
        if row == 0:
            y += SCREEN_SIZE // (field_size * 2)
        else:
            y += (SCREEN_SIZE // (field_size * 2)) * 2
        x = 0
        for col in range(field_size):
            if col == 0:
                x += SCREEN_SIZE // (field_size * 2)
            else:
                x += (SCREEN_SIZE // (field_size * 2)) * 2
            field.append(PointInfo((x, y), field_size))

    pygame.init()
    # 画面サイズ
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    # タイトル
    pygame.display.set_caption("TicTacToe")

    input_state = SquareState.CROSS
    while(True):
        # 画面の塗り潰し
        screen.fill(BACK_GROUND_COLOR)

        # 格子の描画
        separate = SCREEN_SIZE // field_size
        for i in range(1, field_size):
            # 縦線の描画
            pygame.draw.line(screen, LINE_COLOR, (separate * i, PADDING_SIZE), (separate * i, SCREEN_SIZE - PADDING_SIZE), LINE_WIDTH)
            # 横線の描画
            pygame.draw.line(screen, LINE_COLOR, (PADDING_SIZE, separate * i), (SCREEN_SIZE - PADDING_SIZE, separate * i), LINE_WIDTH)

        for f in field:
            if f.state == SquareState.CIRCLE:
                draw_circle(screen, field_size, f.point)
            elif f.state == SquareState.CROSS:
                draw_cross(screen, field_size, f.point)

        # 画面を更新
        pygame.display.update()
        for event in pygame.event.get():
            # クリックしたら座標に記号を入力する
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                for f in field:
                    if f.state == SquareState.NOTHING and x in f.x_range and y in f.y_range:
                        f.state = input_state
                        if input_state == SquareState.CROSS:
                            input_state = SquareState.CIRCLE
                        else:
                            input_state = SquareState.CROSS
                        break

            # バツ押したら終了
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

def draw_circle(screen, field_size, point) -> None:
    radius = (SCREEN_SIZE // (field_size + 1)) / 2.0
    pygame.draw.circle(screen, LINE_COLOR, point, radius, LINE_WIDTH)

def draw_cross(screen, field_size, point) -> None:
    size = (SCREEN_SIZE // (field_size + 1)) / 2.0
    center_x, center_y = point
    pygame.draw.line(screen, LINE_COLOR, (center_x - size, center_y - size), (center_x + size, center_y + size), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (center_x + size, center_y - size), (center_x - size, center_y + size), LINE_WIDTH)

if __name__ == "__main__":
    main()
