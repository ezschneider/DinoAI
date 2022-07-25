import pygame
import os
import random
import time
from sys import exit
import math

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
    def __init__(self, data, state):		
        pass

    def keySelector(self, speed, distance, obHeight):
        pass

    def updateState(self, state):
        pass


def first(x):
    return x[0]


# class KNNClassifier(KeyClassifier):
# 	def __init__(self, data, state):
# 		self.classifier = KNeighborsClassifier(n_neighbors=5)
# 		self.classifier.fit(data, state)

# 	def keySelector(self, speed, distance, obHeight):
# 		label = self.classifier.predict([(speed, distance, obHeight)])
# 		if label == 0:
# 			return "K_DOWN"
# 		elif label == 1:
# 			return "K_UP"
# 		return "K_NO"

# 	def updateState(self, state):
# 		self.state = state


def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame(s):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    # random.seed(3)
    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    cloud = Cloud()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0
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
        print(game_speed)
        distance = 1500
        obHeight = 0
        obType = 2
        if len(obstacles) != 0:
            xy = obstacles[0].getXY()
            distance = xy[0]
            obHeight = obstacles[0].getHeight()
            obType = obstacles[0]

        if GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()
        else:
            aiPlayer = NeuralDinoClassifier(s, n_internalNeurons, n_decisionNeurons)
            userInput = aiPlayer.keySelector(game_speed, distance, obHeight)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        player.update(userInput)
        player.draw(SCREEN)

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
            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(2000)
                death_count += 1
                return points


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
        new_states = [change_state(state, i, ds, 0), change_state(state, i, (-ds), 0), change_state(state, i, 0, dd),
                      change_state(state, i, 0, (-dd))]
        for s in new_states:
            if s != []:
                neighborhood.append(s)
    return neighborhood


# Gradiente Ascent

def gradient_ascent(state, max_time):
    start = time.process_time()
    res, max_value = manyPlaysResults(3)
    better = True
    end = 0
    while better and end - start <= max_time:
        neighborhood = generate_neighborhood(state)
        better = False
        for s in neighborhood:
            aiPlayer = KeySimplestClassifier(s)
            res, value = manyPlaysResults(3)
            if value > max_value:
                state = s
                max_value = value
                better = True
        end = time.process_time()
    return state, max_value


from scipy import stats
import numpy as np


def manyPlaysResults(s, rounds):
    results = []
    for round in range(rounds):
        results += [playGame(s)]
    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())

# ------------------------------------------------------------------------------------------------------- #
# Neural Network Classifier
# ------------------------------------------------------------------------------------------------------- #

class Neuron():
    def __init__(self, weights):
        self.weights = weights
    
    def decision(self, params):
        sum = 0
        for i in range(len(params)):
            sum = sum + self.weights[i]*params[i]
        sum = sum + self.weights[-1]
        return max(0, sum)

# Class of Classification
class NeuralDinoClassifier(KeyClassifier):
    def __init__(self, state, n_internalNeurons, n_decisionNeurons):
        self.state = state
        self.n_internalNeurons = n_internalNeurons
        self.n_decisionNeurons = n_decisionNeurons
        self.interNeurons = []
        self.decisionNeuron = []

    def keySelector(self, distance, obHeight, speed, obType):
        params = np.array([distance, obHeight, speed])
        n_p = len(params)+1
        
        i = 0
        while(i < self.n_internalNeurons):
            n = i * n_p
            p_n = (i + 1) * n_p
            self.interNeurons.append(Neuron(self.state[n:p_n]))
            i=i+1
        i = self.n_internalNeurons
        while(i<self.n_internalNeurons+self.n_decisionNeurons):
            n = i * n_p
            p_n = (i + 1) * n_p
            self.decisionNeuron.append(Neuron(self.state[n:p_n]))
            i=i+1

        sumNeurons = []
        for i in range(self.n_internalNeurons):
            sumNeurons.append(self.interNeurons[i].decision(params))
        
        decision = []
        for i in range(self.n_decisionNeurons):
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
def evaluate_state(rounds, state, n_internalNeurons, n_decisionNeurons):
    aiPlayer = NeuralDinoClassifier(state, n_internalNeurons, n_decisionNeurons)    
    _, value = manyPlaysResults(state, rounds)
    # print("for state: ", state, "value: ", value)
    return value

def states_total_value(states):
    total_sum = 0
    for state in states:
        total_sum = total_sum + state[0]
    return total_sum

def roulette_construction(states):
    aux_states = []
    roulette = []
    total_value = states_total_value(states)

    for state in states:
        value = state[0]
        if total_value != 0:
            ratio = value/total_value
        else:
            ratio = 1
        aux_states.append((ratio,state[1]))

    acc_value = 0
    for state in aux_states:
        acc_value = acc_value + state[0]
        s = (acc_value,state[1])
        roulette.append(s)
    return roulette

def roulette_run (rounds, roulette):
    if roulette == []:
        return []
    selected = []
    while len(selected) < rounds:
        r = random.uniform(0,1)
        for state in roulette:
            if r <= state[0]:
                selected.append(state[1])
                break
    return selected

def generate_initial_state(n_internalNeurons, n_decisionNeurons, params):
    initial_state = []
    min_value = -1000
    max_value = 1000
    for _ in range(params * (n_decisionNeurons+n_internalNeurons)):
        initial_state.append(random.randint(min_value,max_value))
    np_initial_state = np.asarray(initial_state)
    return np_initial_state

def selection(value_population,n):
    aux_population = roulette_construction(value_population)
    new_population = roulette_run(n, aux_population)
    return new_population

def crossover(dad,mom):
    r = random.randint(0, len(dad) - 1)
    son = np.concatenate([dad[:r],mom[r:]])
    daug = np.concatenate([mom[:r],dad[r:]])
    return son, daug

def mutation (indiv):
    min_size = -1000
    max_size = 1000
    individual = indiv.copy()
    n_weights_mutad = 3
    for _ in range(n_weights_mutad):
        rand = random.randint(0, len(individual) - 1)
        individual[rand] = random.randint(min_size,max_size)
    return individual

def initial_population(n, params, n_internalNeurons, n_decisionNeurons):
    pop = []
    count = 0
    while count < n:
        individual = generate_initial_state(n_internalNeurons, n_decisionNeurons, params)
        pop = pop + [individual]
        count += 1
    np_pop = np.asarray(pop)
    return np_pop

def convergent(population):
    if np.array_equal(population, []):
        base = population[0]
        i = 0
        while i < len(population):
            if (base != population[i]).all():
                return False
            i += 1
        return True

def evaluate_population (rounds, pop, n_internalNeurons, n_decisionNeurons):
    eval = []
    for s in pop:
        eval = eval + [(evaluate_state(rounds, s, n_internalNeurons, n_decisionNeurons), s)]
    return eval

def elitism (val_pop, pct):
    n = math.floor((pct/100)*len(val_pop))
    if n < 1:
        n = 1
    val_elite = sorted (val_pop, key = first, reverse = True)[:n]
    elite = [s for v,s in val_elite]
    return elite

def crossover_step (population, crossover_ratio):
    new_pop = []
    
    for _ in range (round(len(population)/2)):
        rand = random.uniform(0, 1)
        fst_ind = random.randint(0, len(population) - 1)
        scd_ind = random.randint(0, len(population) - 1)
        parent1 = population[fst_ind] 
        parent2 = population[scd_ind]

        if rand <= crossover_ratio:
            offspring1, offspring2 = crossover(parent1, parent2)            
            offspring1 = parent1
            offspring2 = parent2
        else:
            offspring1, offspring2 = parent1, parent2
                
        new_pop = new_pop + [offspring1, offspring2]
        
    return new_pop

def mutation_step (population, mutation_ratio):
    ind = 0
    for individual in population:
        rand = random.uniform(0, 1)

        if rand <= mutation_ratio:
            mutated = mutation(individual)
            population[ind] = mutated
                
        ind+=1
        
    return population

def genetic(params, rounds, pop_size, max_iter, cross_ratio, mut_ratio, max_time, elite_pct, n_internalNeurons, n_decisionNeurons):
# mut_ratio, max_time, elite_pct
    global aiPlayers
    aiPlayers = []
    start = time.time()
    opt_state = [0] * params
    opt_value = 0
    pop = initial_population(pop_size, params, n_internalNeurons, n_decisionNeurons)
    conv = convergent(pop)
    iter = 0
    end = 0

    while not conv and iter < max_iter and end-start <= max_time:
        
        val_pop = evaluate_population (rounds, pop, n_internalNeurons, n_decisionNeurons)
        new_pop = elitism (val_pop, elite_pct)
        best = new_pop[0]
        val_best = evaluate_state(rounds, best, n_internalNeurons, n_decisionNeurons)

        if (val_best > opt_value):
            opt_state = best
            opt_value = val_best

        selected = selection(val_pop, pop_size - len(new_pop)) 
        crossed = crossover_step(selected, cross_ratio)
        mutated = mutation_step(crossed, mut_ratio)
        pop = new_pop + mutated
        conv = convergent(pop)
        iter+=1
        end = time.time()
        print(opt_value)


    return opt_state, opt_value, iter, conv

def main():
    global aiPlayer
    random.seed(3)
    params = 4
    rounds = 5
    pop_size = 200
    max_iter = 1000
    cross_ratio = 0.7
    mut_ratio = 0.15
    elite_pct = 10
    max_time = 1000
    n_internalNeurons = 3
    n_decisionNeurons = 2
    # aiPlayer = KNNClassifier(items, initial_state)
    # best_state, best_value = gradient_ascent(initial_state, 5000) 
    # aiPlayer = KeySimplestClassifier(best_state)
    best_state, best_value, iterations, conv = genetic(params, rounds, pop_size, max_iter, cross_ratio, mut_ratio, max_time, elite_pct, n_internalNeurons, n_decisionNeurons)
    # aiPlayer = NeuralDinoClassifier(best_state, n_internalNeurons, n_decisionNeurons)
    res, value = manyPlaysResults(best_state, rounds)
    # playGame()
    # npRes = np.asarray(res)
    # print(res, npRes.mean(), npRes.std(), value)


main()