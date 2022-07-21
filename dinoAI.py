import pygame
from scipy import rand, stats
import numpy as np
import os
import random
import time
import math
from sys import exit

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

    def getXY(self):
        return (self.dino_rect.x, self.dino_rect.y)


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle():
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < - self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return (self.type)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 19:
            self.index = 0
        SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1


class KeyClassifier:
    def __init__(self, state):        
        pass

    def keySelector(self, distance, obHeight, speed, obType):
        pass

    def updateState(self, state):
        pass


def first(x):
    return x[0]

class KeySimplestClassifier(KeyClassifier):
    def __init__(self, state):
        self.state = state

    def keySelector(self, distance, obHeight, speed, obType):
        self.state = sorted(self.state, key=first)
        for s, d in self.state:
            if speed < s:
                limDist = d
                break
        if distance <= limDist:
            if isinstance(obType, Bird) and obHeight > 50:
                return "K_DOWN"
            else:
                return "K_UP"
        return "K_NO"

    def updateState(self, state):
        self.state = state


# ------------------------------------------------------------------------------------------------------- #
# Neural Network Classifier
# ------------------------------------------------------------------------------------------------------- #

# Internal Neuron
class InterNeuron():
    def __init__(self, weights):
        self.distWeight = weights[0]
        self.hWeight = weights[1]
        self.speedWeight = weights[2]
        self.bias = weights[3]

    def decision(self, distance, obHeight, speed):
        sum = self.distWeight*distance + self.hWeight*obHeight + self.speedWeight*speed + self.bias
        if sum >= 0:
            return sum
        else:
            return 0

# External Neuron - Neuron that decides
class DecisionNeuron():
    def __init__(self, weights):
        self.weight0 = weights[0]
        self.weight1 = weights[1]
        self.weight2 = weights[2]
        self.bias = weights[3]
    
    def decision(self, sumNeurons):
        sum = self.weight0*sumNeurons[0] + self.weight1*sumNeurons[1] + self.weight2*sumNeurons[2] + self.bias
        if sum >= 0:
            return sum
        else:
            return 0

# Class of Classification
class NeuralDinoClassifier(KeyClassifier):
    def __init__(self, state):
        self.state = state
        self.interNeurons = []
        self.decisionNeuron = []

    def keySelector(self, distance, obHeight, speed):
        for i in range(3):
            self.interNeurons.append(InterNeuron(self.state[i]))
        for i in range(2):
            self.decisionNeuron.append(DecisionNeuron(self.state[i+3]))

        sumNeurons = []
        for i in range(3):
            sumNeurons.append(self.interNeurons[i].decision(distance, obHeight, speed))
        
        decision = []
        for i in range(2):
            decision.append(self.decisionNeuron[i].decision(sumNeurons))

        if decision[0] > decision[1]:
            return "K_UP"
        elif decision[0] < decision[1]:
            return "K_DOWN"
        else:
            return "K_NO"
        
    def updateState(self, state):
        self.state = state

# ------------------------------------------------------------------------------------------------------- #
# Genetic Algorithm
# ------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------- #

def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True
    clock = pygame.time.Clock()
    players = []
    for i, val in enumerate(aiPlayers):        
        players.append((Dinosaur(),i))
    cloud = Cloud()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0
    results = []
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    death_count = 0
    spawn_dist = 0

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        SCREEN.fill((255, 255, 255))

        distance = 1500
        obHeight = 0
        obType = 2
        if len(obstacles) != 0:
            xy = obstacles[0].getXY()
            distance = xy[0]
            obHeight = obstacles[0].getHeight()
            obType = obstacles[0]

        userInput = []
        # if GAME_MODE == "HUMAN_MODE":
        #     userInput[0] = playerKeySelector()
        # else:
        for i, val in enumerate(players):
            userInput.append(aiPlayers[i].keySelector(distance, obHeight, game_speed, obType))

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        for i, player in enumerate(players):
            player[0].update(userInput[i])
            player[0].draw(SCREEN)

        for obstacle in list(obstacles):
            obstacle.update()
            obstacle.draw(SCREEN)

        background()

        cloud.draw(SCREEN)
        cloud.update()

        score()

        clock.tick(60)
        pygame.display.update()

        for obstacle in obstacles:
            for i, player in enumerate(players):
                if player[0].dino_rect.colliderect(obstacle.rect):
                    results.append((points, player[1]))
                    if len(players) == 1:
                        pygame.time.delay(2000)
                        death_count += 1
                        return sorted(results, key=lambda x: x[1])
                    players.remove(player)


# Change State Operator

def change_state(state, position, vs, vd):
    aux = state.copy()
    s, d = state[position]
    ns = s + vs
    nd = d + vd
    if ns < 15 or nd > 1000:
        return []
    return aux[:position] + [(ns, nd)] + aux[position + 1:]


# Neighborhood

def generate_neighborhood(state):
    neighborhood = []
    state_size = len(state)    
    for i in range(state_size):
        ds = random.randint(1, 10) 
        dd = random.randint(1, 100) 
        new_states=[change_state(state, i, ds, 0), change_state(state, i, (-ds), 0), change_state(state, i, 0, dd),
                    change_state(state, i, 0, (-dd))]
        for s in new_states:
            if s != []:
                neighborhood.append(s)
    return neighborhood


# Gradiente Ascent

def gradient_ascent(state, max_time):
    start = time.process_time()
    max_value, _ = manyPlaysResults(3)
    better = True
    end = 0
    while better and end - start <= max_time:
        neighborhood = generate_neighborhood(state)
        better = False
        global aiPlayers
        aiPlayers = []
        for s in neighborhood:            
            aiPlayers.append(NeuralDinoClassifier(s))        
        # print(aiPlayers)        
        value, index = manyPlaysResults(3)
        if value > max_value:
            print("New max value: " + str(value))            
            state = neighborhood[index]
            print("New state: " + str(state))
            max_value = value
            better = True
        end = time.process_time()
    return state, max_value


def manyPlaysResults(rounds):
    results = []
    for round in range(rounds):
        results += [playGame()]
    scores = []
    for i, result in enumerate(results):
        for j, val in enumerate(result):
            if i == 0:
                scores.append([val[0]])
            else:
                scores[j].append(val[0])
    npResults = np.asarray(results)
    scores = np.asarray(scores)
    scores = [x.mean() - x.std() for x in scores]
    return (max(scores), scores.index(max(scores)))

def creatInitial_state(x, y):
                    # State of Internal Neurons
    initial_state = [[rand(x, y), rand(x, y), rand(x, y), rand(x, y)],
                    [rand(x, y), rand(x, y), rand(x, y), rand(x, y)],
                    [rand(x, y), rand(x, y), rand(x, y), rand(x, y)],
                    # State of Decision Neurons
                    [rand(x, y), rand(x, y), rand(x, y), rand(x, y)],
                    [rand(x, y), rand(x, y), rand(x, y), rand(x, y)],]
    return initial_state

def main():
    global aiPlayers
    aiPlayers = []
    # initial_state = [(15, 250), (18, 350), (20, 450), (1000, 550)]
    for _ in range(10):
        initial_state = creatInitial_state(-1000,1000)
        aiPlayers.append(NeuralDinoClassifier(initial_state))
    best_state, best_value = manyPlaysResults(3)
    print("Best state: " + str(best_state))
    print("Best value: " + str(best_value))
    aiPlayers = []
    aiPlayers.append(KeySimplestClassifier(best_state))
    res, value = manyPlaysResults(3)
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)


main()