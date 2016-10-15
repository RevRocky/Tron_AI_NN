import pygame, random
from NEAT_TRON import neatNN
import numpy as np

size = width, height = 160, 160 #must be multiple of four due to how the randomized spawning works
black = 0,0,0
dark_blue = 0,0,255
light_blue = 0,255,255
htot = 0
atot = 0
ttot = 0
running = True
human_play = True
board = np.array(0)
board.resize(size)
board.fill(2)

def add(a, b):
    return (a[0]+b[0], a[1]+b[1])
def sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

def oob(p, w, h):
    if p[0] >= w or p[0] < 0 or p[1] >= h or p[1] < 0:
        return True
    return False

def sign(n):
    if n > 0: return 1
    if n < 0: return -1
    return 0

def getBoard():
    return board

while running:
    pygame.init()

    display = pygame.display.set_mode((width * 4, height * 4))
    pygame.display.set_caption('Tron')
    screen = pygame.Surface(size)
    screen.fill(black)

    clock = pygame.time.Clock()

    d1 = 1,0

    p1Points = []
    p2Points = []

    mindiff = 15
    while True:
        pos1 = random.randint(width / 4, 3 * (width / 4)), random.randint(height / 4, 3 * (height / 4))
        pos2 = random.randint(width / 4, 3 * (width / 4)), random.randint(height / 4, 3 * (height / 4))
        if not(abs(pos1[0] - pos2[0]) < mindiff or abs(pos1[1] - pos2[1]) < mindiff):
            board[pos1[0]][pos1[1]] = 1
            board[pos2[0]][pos2[1]] = 0
            break

    ai_lose = False
    human_lose = False

    playing = True
    while playing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                playing = False
            if event.type == pygame.KEYDOWN:
                if event.key==pygame.K_w:
                    d1 = 0, -1
                elif event.key==pygame.K_s:
                    d1 = 0, 1
                elif event.key==pygame.K_a:
                    d1 = -1, 0
                elif event.key==pygame.K_d:
                    d1 = 1, 0

        aim = x, y = sub(add(add(pos1, d1), d1), pos2)
        if abs(x) > abs(y):
            optimal = sign(x), 0
            optimal2 = 0, sign(y)
        else:
            optimal = 0, sign(y)
            optimal2 = sign(x), 0
        if (not oob(add(pos2, optimal), width, height)) and screen.get_at(add(pos2, optimal)) == black:
           d2 = optimal
        elif (not oob(add(pos2, optimal2), width, height)) and screen.get_at(add(pos2, optimal2)) == black:
            d2 = optimal2
        elif (not oob(add(pos2, (optimal2[0] * -1, optimal2[1] * -1)), width, height)) and screen.get_at(add(pos2, (optimal2[0] * -1, optimal2[1] * -1))) == black:
            d2 = optimal2[0] * -1, optimal2[1] * -1
        elif (not oob(add(pos2, (optimal[0] * -1, optimal[1] * -1)), width, height)) and screen.get_at(add(pos2, (optimal[0] * -1, optimal[1] * -1))) == black:
            d2 = optimal[0] * -1, optimal[1] * -1

        board[pos1[0]][pos1[1]] = -1
        pos1 = add(pos1, d1)
        pos2 = add(pos2, d2)
        board[pos1[0]][pos1[1]] = 1
        board[pos2[0]][pos2[1]] = 0

        p1Points.append(pos1)
        p2Points.append(pos2)

        if pos1 == pos2:
            ai_lose = True
            human_lose = True
            ttot += 1
            playing = False
        else:
            if oob(pos1, width, height) or screen.get_at(pos1) != black:
                ai_lose = True
                atot += 1
                playing = False
            else:
                screen.set_at(pos1, (0, 255, 255))
                if len(p1Points) > 1:
                    screen.set_at(p1Points[-2], (0, 0, 255))
            if oob(pos2, width, height) or screen.get_at(pos2) != black:
                human_lose = True
                htot += 1
                playing = False
            else:
                screen.set_at(pos2, (255, 255, 0))
                if len(p2Points) > 1:
                    screen.set_at(p2Points[-2], (255, 0, 0))

        neatNN.increaseFitness(1)
        clock.tick(25)

        pygame.transform.scale(screen, (width * 4, height * 4), display)

        pygame.display.flip()

    pygame.quit()

    if ai_lose and human_lose:
        print('Tie')
        neatNN.nextGenome()
    elif ai_lose:
        print('AI Wins!')
        neatNN.increaseFitness(400)
        neatNN.nextGenome()
    elif human_lose:
        print('You Win')
        neatNN.nextGenome()

    res = ''
    while res not in ['y', 'n', 'Y', 'N', '']:
        if human_lose:
            res = str(input("Rematch? Please? [Y, n]"))
        else:
            res = str(input("Would you like a rematch? [Y, n]"))
    if res in ['n', 'N']:
        running = False

print ('You won', htot, 'times, the AI won', atot, 'times, and you tied', ttot, 'times. Good Game.')
