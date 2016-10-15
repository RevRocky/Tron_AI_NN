import pygame, random
from NEAT_TRON import neatNN
import numpy as np

size = width, height = 160, 160 #must be multiple of four due to how the randomized spawning works
opponentWinCount = 0
neuralNetworkWinCount = 0
tieCount = 0
running = True
humanPlayer = True
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

    minimumDifference = 15
    while True:
        opponentPosition = random.randint(width / 4, 3 * (width / 4)), random.randint(height / 4, 3 * (height / 4))
        neuralNetworkPosition = random.randint(width / 4, 3 * (width / 4)), random.randint(height / 4, 3 * (height / 4))
        if not(abs(opponentPosition[0] - neuralNetworkPosition[0]) < minimumDifference or abs(opponentPosition[1] - neuralNetworkPosition[1]) < minimumDifference):
            board[opponentPosition[0]][opponentPosition[1]] = 1
            board[neuralNetworkPosition[0]][neuralNetworkPosition[1]] = 0
            break

    neuralNetworkLost = False
    opponentLost = False

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

        neatNN.evaluate(board, neuralNetworkPosition)

        aim = x, y = sub(add(add(opponentPosition, d1), d1), neuralNetworkPosition)
        if abs(x) > abs(y):
            optimal = sign(x), 0
            optimal2 = 0, sign(y)
        else:
            optimal = 0, sign(y)
            optimal2 = sign(x), 0
        if (not oob(add(neuralNetworkPosition, optimal), width, height)) and screen.get_at(add(neuralNetworkPosition, optimal)) == black:
           d2 = optimal
        elif (not oob(add(neuralNetworkPosition, optimal2), width, height)) and screen.get_at(add(neuralNetworkPosition, optimal2)) == black:
            d2 = optimal2
        elif (not oob(add(neuralNetworkPosition, (optimal2[0] * -1, optimal2[1] * -1)), width, height)) and screen.get_at(add(neuralNetworkPosition, (optimal2[0] * -1, optimal2[1] * -1))) == black:
            d2 = optimal2[0] * -1, optimal2[1] * -1
        elif (not oob(add(neuralNetworkPosition, (optimal[0] * -1, optimal[1] * -1)), width, height)) and screen.get_at(add(neuralNetworkPosition, (optimal[0] * -1, optimal[1] * -1))) == black:
            d2 = optimal[0] * -1, optimal[1] * -1

        board[opponentPosition[0]][opponentPosition[1]] = -1
        opponentPosition = add(opponentPosition, d1)
        neuralNetworkPosition = add(neuralNetworkPosition, d2)
        board[opponentPosition[0]][opponentPosition[1]] = 1
        board[neuralNetworkPosition[0]][neuralNetworkPosition[1]] = 0

        p1Points.append(opponentPosition)
        p2Points.append(neuralNetworkPosition)

        if opponentPosition == neuralNetworkPosition:
            neuralNetworkLost = True
            opponentLost = True
            tieCount += 1
            playing = False
        else:
            if oob(opponentPosition, width, height) or screen.get_at(opponentPosition) != black:
                neuralNetworkLost = True
                neuralNetworkWinCount += 1
                playing = False
            else:
                screen.set_at(opponentPosition, (0, 255, 255))
                if len(p1Points) > 1:
                    screen.set_at(p1Points[-2], (0, 0, 255))
            if oob(neuralNetworkPosition, width, height) or screen.get_at(neuralNetworkPosition) != black:
                opponentLost = True
                opponentWinCount += 1
                playing = False
            else:
                screen.set_at(neuralNetworkPosition, (255, 255, 0))
                if len(p2Points) > 1:
                    screen.set_at(p2Points[-2], (255, 0, 0))

        clock.tick(25)

        pygame.transform.scale(screen, (width * 4, height * 4), display)

        pygame.display.flip()

    pygame.quit()

    if neuralNetworkLost and opponentLost:
        print('Tie')
        neatNN.nextGenome(False)
    elif neuralNetworkLost:
        print('AI Wins!')
        neatNN.nextGenome(True)
    elif opponentLost:
        print('You Win')
        neatNN.nextGenome(False)

    response = ''
    while response not in ['y', 'n', 'Y', 'N', '']:
        if opponentLost:
            response = str(input("Rematch? Please? [Y, n]"))
        else:
            response = str(input("Would you like a rematch? [Y, n]"))
    if response in ['n', 'N']:
        running = False

print ('You won', opponentWinCount, 'times, the AI won', neuralNetworkWinCount, 'times, and you tied', tieCount, 'times. Good Game.')
