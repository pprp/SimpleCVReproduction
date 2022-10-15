# https://github.com/haroldsultan/MCTS
import random
import math
import hashlib
import logging
import argparse


"""
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf
The State is a game where you have NUM_TURNS and at turn i you can make
a choice from an integeter [-2,2,3,-3]*(NUM_TURNS+1-i).  So for example in a game of 4 turns, on turn for turn 1 you can can choose from [-8,8,12,-12], and on turn 2 you can choose from [-6,6,9,-9].  At each turn the choosen number is accumulated into a aggregation value.  The goal of the game is for the accumulated value to be as close to 0 as possible.
The game is not very interesting but it allows one to study MCTS which is.  Some features 
of the example by design are that moves do not commute and early mistakes are more costly.  
In particular there are two models of best child that one can use 
"""

# MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
SCALAR = 1/(2*math.sqrt(2.0))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')


class State():
    NUM_TURNS = 10
    GOAL = 0
    MOVES = [2, -2, 3, -3]
    MAX_VALUE = (5.0*(NUM_TURNS-1)*NUM_TURNS)/2
    num_moves = len(MOVES)

    def __init__(self, value=0, moves=None, turn=NUM_TURNS):
        if moves is None:
            moves = []
        self.value = value
        self.turn = turn
        self.moves = moves

    def next_state(self):
        nextmove = random.choice([x*self.turn for x in self.MOVES])
        return State(self.value+nextmove, self.moves+[nextmove], self.turn-1)

    def terminal(self):
        return self.turn == 0

    def reward(self):
        return 1.0-(abs(self.value-self.GOAL)/self.MAX_VALUE)

    def __hash__(self):
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return "Value: %d; Moves: %s" % (self.value, self.moves)


class Node():
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self, num_moves_lambda):
        if num_moves_lambda is None:
            num_moves = self.state.num_moves
        else:
            num_moves = num_moves_lambda(self)
        return len(self.children) == num_moves

    def __repr__(self):
        return "Node; children: %d; visits: %d; reward: %f" % (len(self.children), self.visits, self.reward)


def UCTSEARCH(budget, root, num_moves_lambda=None):
    for iter in range(int(budget)):
        if iter % 10000 == 9999:
            logger.info("simulation: %d" % iter)
            logger.info(root)
        front = TREEPOLICY(root, num_moves_lambda)
        reward = DEFAULTPOLICY(front.state)
        BACKUP(front, reward)
    return BESTCHILD(root, 0)


def TREEPOLICY(node, num_moves_lambda):
    # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.state.terminal() == False:
        if len(node.children) == 0:
            return EXPAND(node)
        elif random.uniform(0, 1) < .5:
            node = BESTCHILD(node, SCALAR)
        elif node.fully_expanded(num_moves_lambda) == False:
            return EXPAND(node)
        else:
            node = BESTCHILD(node, SCALAR)
    return node


def EXPAND(node):
    tried_children = [c.state for c in node.children]
    new_state = node.state.next_state()
    while new_state in tried_children and new_state.terminal() == False:
        new_state = node.state.next_state()
    node.add_child(new_state)
    return node.children[-1]

# current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)


def BESTCHILD(node, scalar):
    bestscore = 0.0
    bestchildren = []
    for c in node.children:
        exploit = c.reward/c.visits
        explore = math.sqrt(2.0*math.log(node.visits)/float(c.visits))
        score = exploit+scalar*explore
        if score == bestscore:
            bestchildren.append(c)
        if score > bestscore:
            bestchildren = [c]
            bestscore = score
    if len(bestchildren) == 0:
        logger.warn("OOPS: no best child found, probably fatal")
    return random.choice(bestchildren)


def DEFAULTPOLICY(state):
    while state.terminal() == False:
        state = state.next_state()
    return state.reward()


def BACKUP(node, reward):
    while node != None:
        node.visits += 1
        node.reward += reward
        node = node.parent
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCTS research code')
    parser.add_argument('--num_sims', action="store", required=True, type=int)
    parser.add_argument('--levels', action="store", required=True,
                        type=int, choices=range(State.NUM_TURNS+1))
    args = parser.parse_args()

    current_node = Node(State())
    for l in range(args.levels):
        current_node = UCTSEARCH(args.num_sims/(l+1), current_node)
        print("level %d" % l)
        print("Num Children: %d" % len(current_node.children))
        for i, c in enumerate(current_node.children):
            print(i, c)
        print(f"Best Child: {current_node.state}")

        print("--------------------------------")
