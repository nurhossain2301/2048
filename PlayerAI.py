from random import randint
from BaseAI import BaseAI
import math
import time
import sys

startTime = 0
class PlayerAI(BaseAI):
    def __init__(self):
        self.timeLimit = 0.18

    def getMove(self, grid):
        global startTime
        startTime = time.clock()
        depth =1
        max_utility = -math.inf
        while True:
            moves, utility, over = self.maximize_alpha(grid, depth, -math.inf, math.inf)
            if over:
                break
            if utility > max_utility:
                move, max_utility = moves, utility
            depth += 1
        return move

    #minimax algorithm with alpha beta pruning
    def minimize_beta(self, grid, depth, alpha, beta):
        if time.clock() - startTime > self.timeLimit:
            return -1, self.utility(grid), True
        if depth == 0:
            return None, self.utility(grid), False
        (minChild, minUtility) = (None, beta)
        cells = grid.getAvailableCells()
        if len(cells) ==0:
            return None, self.utility(grid), False
        for cell in cells:
            new_grid = grid.clone()
            new_grid.setCellValue(cell, 2)
            (child, utility, over) = self.maximize_alpha(new_grid, depth-1, alpha, beta)
            if utility < minUtility:
                minUtility = utility
            if minUtility <= alpha:
                break
            if minUtility < beta:
                beta = minUtility
        return None, minUtility, False
    
    def maximize_alpha(self, grid, depth, alpha, beta):
        if time.clock() - startTime > self.timeLimit:
            return -1, self.utility(grid), True
        if depth == 0:
            return -1, self.utility(grid), False
        (maxChild, maxUtility) = (None, alpha)
        children = grid.getAvailableMoves()
        if len(children) ==0:
            return None, self.utility(grid), False
        for child in children:
            new_grid = grid.clone()
            new_grid.move(child)
            (_, utility, over) = self.minimize_beta(new_grid, depth-1, alpha, beta)
            if utility > maxUtility:
                (maxChild, maxUtility) = (child, utility)
            if maxUtility >= beta:
                break
            if maxUtility > alpha:
                alpha = maxUtility
        return (maxChild, maxUtility, False)

    #utility function
    def utility(self, grid):
        abs_val = self.smoothness(grid)
        mono = self.monotony(grid)
        emptyCell = math.log(len(grid.getAvailableCells())+1)
        max = math.log(grid.getMaxTile()) * len(grid.getAvailableCells())/ math.log(2)
        edge = self.edge_val(grid, grid.getMaxTile())
        return mono + 0.1 * abs_val + 2.7* emptyCell + max + 10 * edge

    @staticmethod
    def get_max_value(max_tile, empty_cells):
        return math.log(max_tile) * empty_cells / math.log(2)

    #heuristic edge value
    @staticmethod
    def edge_val(grid, maxTile):
        edge = 0

        for x in range(4):
            for y in range(4):
                if maxTile == grid.getCellValue((x,y)):
                    if maxTile < 1024:
                        edge = -((abs(x - 0) + abs(y - 0)) * maxTile)
                    else:
                        edge = -((abs(x - 0) + abs(y - 0)) * (maxTile / 2))
                    break
        return edge

    #heuristic smoothness
    @staticmethod
    def smoothness(grid):
        cells = grid.getAvailableCells()
        abs_val = 0
        for i in range(4):
            for j in range(4):
                if (i, j) not in cells:
                    cell = math.log(grid.getCellValue((i, j))) / math.log(2)
                    if i < 3:
                        if (i + 1, j) not in cells:
                            n1 = math.log(grid.getCellValue((i + 1, j))) / math.log(2)
                            abs_val -= abs(cell - n1)
                    if j < 3:
                        if (i, j + 1) not in cells:
                            n2 = math.log(grid.getCellValue((i, j + 1))) / math.log(2)
                            abs_val -= abs(cell - n2)

        return abs_val

    #heuristic monotony
    @staticmethod
    def monotony(grid):
        monotomy = [0, 0, 0, 0]

        for x in range(4):
            cell = 0
            neighbor = cell + 1
            while neighbor < 4:
                value = 0 if grid.getCellValue((x, cell)) == 0 else math.log(grid.getCellValue((x, cell)))
                n = 0 if grid.getCellValue((x, neighbor)) == 0 else math.log(grid.getCellValue((x, neighbor)))
                if value > n:
                    monotomy[0] += (value + n)
                elif n > value:
                    monotomy[1] += n-value
                cell = neighbor
                neighbor +=1
        for y in range(4):
            cell = 0
            neighbor = cell + 1
            while neighbor < 4:
                value = 0 if grid.getCellValue((cell, y)) == 0 else math.log(grid.getCellValue((cell, y)))/math.log(2)
                n = 0 if grid.getCellValue((neighbor, y)) == 0 else math.log(grid.getCellValue((neighbor, y)))/math.log(2)
                if value > n:
                    monotomy[2] += (n - value)
                elif n > value:
                    monotomy[3] += value -n
                cell = neighbor
                neighbor +=1
        mono = max(monotomy[0], monotomy[1]) + max(monotomy[2], monotomy[3])
        return mono
