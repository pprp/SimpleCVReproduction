# https://github.com/haroldsultan/MCTS
from pydoc import describe
import random
import math
import hashlib
import logging
import argparse

from typing import List

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
    """_summary_

        Args:
            NUM_TRUNS: left depth of network.
            GOAL: close to 0 is better 
            MOVES: choices provided
            MAX_VALUE: used to normalize the reward score 
            num_moves: number of choices
            self.moves: current choices list 
            self.value: current status
    """
    NUM_TURNS = 10
    GOAL = 0
    MOVES = [2, -2, 3, -3]
    MAX_VALUE = (5.0*(NUM_TURNS-1)*NUM_TURNS)/2
    num_moves = len(MOVES)

    def __init__(self, value: int = 0, moves: List = None, turn: int = NUM_TURNS):
        if moves is None:
            moves = []
        self.value = value
        self.turn = turn
        self.moves = moves

    def next_state(self):
        """generate next states by current node"""
        nextmove = random.choice([x*self.turn for x in self.MOVES])
        return State(self.value+nextmove, self.moves+[nextmove], self.turn-1)

    def terminal(self):
        """terminate when move the leaf node."""
        return self.turn == 0

    def reward(self):
        """reward functions."""
        return 1.0-(abs(self.value-self.GOAL)/self.MAX_VALUE)

    def __hash__(self):
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return "Value: %d; Moves: %s" % (self.value, self.moves)


class Node():
    """Node of MC Tree.

    Args:
        state (_type_): states of current node.
        parent (_type_, optional): parenet node of current node. 
            Defaults to None.
    """

    def __init__(self, state, parent=None):
        self.visits = 1  # number of visited, used for normalize UTC score
        self.reward = 0.0  # reward of current node
        self.state = state  # states of current node
        self.children = []  # list of children node
        self.parent = parent  # record parent node

    def add_child(self, child_state):
        """add child node for current node."""
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        """No function called.??"""
        self.reward += reward
        self.visits += 1

    def fully_expanded(self, num_moves_lambda):
        """judge whether currnet node is fully expanded."""
        # num_moves means number of choices.
        if num_moves_lambda is None:
            num_moves = self.state.num_moves
        else:
            num_moves = num_moves_lambda(self)
        return len(self.children) == num_moves

    def __repr__(self):
        return f"Node; children: {len(self.children)}; visits: {self.visits}; reward: {self.reward}"


def UCTSEARCH(budget, root, num_moves_lambda=None):
    """Main search process of UCT

    Args:
        budget: the number of simulations to perform
        root: root node of MC tree
    """
    for iter in range(int(budget)):
        if iter % 10000 == 9999:
            logger.info("simulation: %d" % iter)
            logger.info(root)

        front = freepolicy(root, num_moves_lambda)
        reward = default_policy(front.state)
        backup(front, reward)

    return best_child(root, 0)


def freepolicy(node, num_moves_lambda):
    # a hack to force 'exploitation' in a game where there are many options, 
    # and you may never/not want to fully expand_node first
    while node.state.terminal() == False:
        if len(node.children) == 0:
            # expand_node all of child node for root node
            return expand_node(node)
        elif random.uniform(0, 1) < .5:
            # exploitation with random policy for middle node
            # 50% get the best_child
            node = best_child(node, SCALAR)
        elif node.fully_expanded(num_moves_lambda) == False:
            # not fully explored nodes for middle node.
            return expand_node(node)
        else:
            # leaf node.
            node = best_child(node, SCALAR)
    return node


def expand_node(node):
    """expand one child node for current node."""
    tried_children = [c.state for c in node.children]
    # get children generated by `next_state`
    new_state = node.state.next_state()
    while new_state in tried_children and new_state.terminal() == False:
        new_state = node.state.next_state()
    # append the new node to current node.
    node.add_child(new_state)
    return node.children[-1]

# current this uses the most vanilla MCTS formula
# it is worth experimenting with THRESHOLD ASCENT (TAGS)
def best_child(node, scalar):
    """get best child by UTC score."""
    bestscore = 0.0
    best_children = []
    for c in node.children:
        exploit = c.reward / c.visits
        explore = math.sqrt(2.0*math.log(node.visits)/float(c.visits))
        # UTC score calculation
        score = exploit + scalar*explore
        if score == bestscore:
            best_children.append(c)
        if score > bestscore:
            best_children = [c]
            bestscore = score
    if len(best_children) == 0:
        logger.warn("OOPS: no best child found, probably fatal")
    return random.choice(best_children)


def default_policy(state):
    """Judge whether to terminate."""
    while state.terminal() == False:
        state = state.next_state()
    return state.reward()


def backup(node, reward):
    """back propagation and update the reward of nodes."""
    while node != None:
        node.visits += 1
        node.reward += reward
        node = node.parent
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCTS research code')
    parser.add_argument('--num_sims', action="store", required=True,
                        type=int, help="the number of simulations to perform")
    parser.add_argument('--levels', action="store", required=True,
                        type=int, choices=range(State.NUM_TURNS+1), 
                        help="the number of times to use MCTS to pick a best child")
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
