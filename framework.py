import random
import time
import numpy as np
from neatNN import neatNN
import saveToFile

MAX_GEN = -1 #-1 is unlimited
SIZE = 160,160
FPS = 0 #Below 1 is unlimited
HUMAN = False
PYGAME = False
ADV_DBG = True
LOAD_FROM_FILE = False
SAVE_TO_FILE = True
PLAY = False
LOAD_FILE_NAME = "save"
SAVE_FILE_NAME = "save"
BEST_FILE_NAME = "best"
PLAY_FILE_NAME = "best"
STAT_FILE_NAME = "stats"

if PYGAME:
  import pygame

class Tron(object):
  def __init__(self, size):
    self.size = size
    if PYGAME:
      pygame.init()
      self.d = pygame.display.set_mode((self.size[0] * 4,self.size[1]*4))
      self.s = pygame.Surface(self.size)
    self.quitFlag = False
    self.toErase = []

  def start(self):
    self.board = np.empty(self.size)
    self.board.fill(2)
    w = self.size[0]
    h = self.size[1]
    minDif = 15
    while True:
      self.p2 = random.randint(w / 4, 3 * (w / 4)), random.randint(h / 4, 3 * (h / 4))
      self.p1 = random.randint(w / 4, 3 * (w / 4)), random.randint(h / 4, 3 * (h / 4))
      if not(abs(self.p2[0] - self.p1[0]) < minDif or abs(self.p2[1] - self.p1[1]) < minDif):
        break
    self.d1 = (1, 0)
    self.d2 = (-1, 0)
    if PYGAME:
      self.s.fill((0,0,0))

  def quit(self):
    if PYGAME:
      pygame.display.quit()
      pygame.quit()

  def oob(self, p, w, h):
    if p[0] >= w or p[0] < 0 or p[1] >= h or p[1] < 0:
      return True
    return False

  def tick(self, move, genome):
    for d in self.toErase:
      self.s.set_at(d[0], d[1])
    self.toErase = []

    self.oldd1 = self.d1
    if move == "U":
      self.d1 = (0, -1)
    elif move == "D":
      self.d1 = (0, 1)
    elif move == "L":
      self.d1 = (-1, 0)
    elif move == "R":
      self.d1 = (1, 0)

    if PYGAME:
      for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and HUMAN:
          if event.key==pygame.K_w:
            self.d2 = 0, -1
          elif event.key==pygame.K_s:
            self.d2 = 0, 1
          elif event.key==pygame.K_a:
            self.d2 = -1, 0
          elif event.key==pygame.K_d:
            self.d2 = 1, 0
        if event.type == pygame.QUIT:
          self.quitFlag = True

    if not HUMAN or not PYGAME:
      self.d2 = ai(self.board, self.p1, self.p2, self.oldd1, self.d2, self.size[0], self.size[1])

    oldp2 = self.p2

    self.p1 = add(self.p1, self.d1)
    self.p2 = add(self.p2, self.d2)

    self.p1 = int(self.p1[0]), int(self.p1[1])
    self.p2 = int(self.p2[0]), int(self.p2[1])

    if self.p1 == self.p2:
      return self.p1, (0, 1)
    else:
      if self.oob(self.p1, self.size[0], self.size[1]) or self.board[self.p1[0]][self.p1[1]] != 2:
        return self.p1, (0, 1)
      elif PYGAME:
        self.s.set_at(self.p1, (0, 0, 255))
      if self.oob(self.p2, self.size[0], self.size[1]) or self.board[self.p2[0]][self.p2[1]] != 2:
        return self.p1, (1, 0)
      elif PYGAME:
        self.s.set_at(self.p2, (255, 0, 0))

    self.board[self.p1[0]][self.p1[1]] = 0
    self.board[self.p2[0]][self.p2[1]] = 1
    self.board[oldp2[0]][oldp2[1]] = -1

    if PYGAME and ADV_DBG:
      for n in genome.nodes:
        if n.nType == "in":
          p = n.pos[0] - (self.size[0] - 1) + self.p1[0], n.pos[1] - (self.size[1] - 1) + self.p1[1]
          if not self.oob(p, self.size[0], self.size[1]):
            self.toErase.append((p, self.s.get_at(p)))
            self.s.set_at(p, (0, 255, 0))

    if PYGAME:
      pygame.transform.scale(self.s, (self.size[0] * 4, self.size[1] * 4), self.d)
      pygame.display.flip()

    if FPS > 0:
      time.sleep(1 / FPS)

    return self.p1, (0,0)

def choice(gen1, gen2):
  num = random.randint(0, 10)
  if num % 2 == 0:
    return gen1
  else:
    return gen2

def add(a, b):
  res = a[0]+b[0], a[1]+b[1]
  return res

def sub(a, b):
  res = a[0]-b[0], a[1]-b[1]
  return res

def oob(p, w, h):
  if p[0] >= w or p[0] < 0 or p[1] >= h or p[1] < 0:
    return True
  return False

def sign(n):
  if n > 0:
    return 1
  if n < 0:
    return -1
  return 0

def ai(board, opPos, yourPos, d1, d2, width, height): #Sorry for the messy AI code
  x, y = sub(add(add(opPos, d1), d1), yourPos)
  if abs(x) > abs(y):
    optimal = sign(x), 0
    optimal2 = 0, sign(y)
  else:
    optimal = 0, sign(y)
    optimal2 = sign(x), 0
  if (not oob(add(yourPos, optimal), width, height)) and board[add(yourPos, optimal)[0]][add(yourPos, optimal)[1]] == 2:
    d2 = optimal
  elif (not oob(add(yourPos, optimal2), width, height)) and board[add(yourPos, optimal2)[0]][add(yourPos, optimal2)[1]] == 2:
    d2 = optimal2
  elif (not oob(add(yourPos, (optimal2[0] * -1, optimal2[1] * -1)), width, height)) and board[add(yourPos, (optimal2[0] * -1, optimal2[1] * -1))[0]][add(yourPos, (optimal2[0] * -1, optimal2[1] * -1))[0]] == 2:
    d2 = optimal2[0] * -1, optimal2[1] * -1
  elif (not oob(add(yourPos, (optimal[0] * -1, optimal[1] * -1)), width, height)) and board[add(yourPos, (optimal[0] * -1, optimal[1] * -1))[0]][add(yourPos, (optimal[0] * -1, optimal[1] * -1))[0]] == 2:
    d2 = optimal[0] * -1, optimal[1] * -1

  return d2

def main():
  if SAVE_TO_FILE:
    saveToFile.clear_file(SAVE_FILE_NAME, STAT_FILE_NAME)

  framework = Tron(SIZE)
  trainingNets = neatNN(framework)
  if LOAD_FROM_FILE:
    trainingNets.start(saveToFile.readPickledFile(LOAD_FILE_NAME))
  else:
    trainingNets.start()
  generation = 0
  while generation < MAX_GEN or MAX_GEN == -1:
    startTime = time.time()
    print("Generation", generation, "begin")
    maxF, avgF, save, best = trainingNets.doGeneration()
    if best is None:
      break
    endTime = time.time()
    if SAVE_TO_FILE:
      saveToFile.record(generation, endTime - startTime, avgF, maxF, STAT_FILE_NAME)
      saveToFile.pickleToDisk(save, SAVE_FILE_NAME)
      saveToFile.pickleToDisk(best, BEST_FILE_NAME)
    generation += 1

  framework.quit()

def play():
  global ADV_DBG
  framework = Tron(SIZE)
  trainingNets = neatNN(framework)
  PYGAME = True
  FPS = 30
  ADV_DBG = False
  trainingNets.evaluate(saveToFile.readPickledFile(PLAY_FILE_NAME))

if not PLAY:
  main()
else:
  play()
