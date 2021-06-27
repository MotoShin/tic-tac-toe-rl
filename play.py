import pygame
from pygame.locals import *
import sys
import math

SCREEN_SIZE = 800
PADDING_SIZE = 10

def main():
    field_size = 3

    pygame.init()
    # 画面サイズ
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    # タイトル
    pygame.display.set_caption("TicTacToe")

    while(True):
        # 画面の塗り潰し
        screen.fill((0, 0, 0))

        # 格子の描画
        separate = SCREEN_SIZE // field_size
        for i in range(1, field_size):
            # 縦線の描画
            pygame.draw.line(screen, (255, 255, 255), (separate * i, PADDING_SIZE), (separate * i, SCREEN_SIZE - PADDING_SIZE), 5)
            # 横線の描画
            pygame.draw.line(screen, (255, 255, 255), (PADDING_SIZE, separate * i), (SCREEN_SIZE - PADDING_SIZE, separate * i), 5)

        radius = (SCREEN_SIZE // (field_size + 1)) / 2.0
        pygame.draw.circle(screen, (255, 255, 255), (400, 400), radius, 5)

        # size = SCREEN_SIZE // (field_size + 1)
        # harf_diagonal = size / 2.0 * math.sqrt(2)
        # center_x = 400
        # center_y = 400
        # pygame.draw.line(screen, (255, 255, 255),
        #     (center_x - harf_diagonal, center_y - harf_diagonal),
        #     (center_x + harf_diagonal, center_y + harf_diagonal), 5)


        # 画面を更新
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    main()
