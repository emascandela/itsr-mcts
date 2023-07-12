import numpy as np
from typing import Tuple


from models.preprocessors.preprocessor import Preprocessor


class Node:
    def __init__(
        self,
        parent: "Node" = None,
        action: int = None,
        prior_prob: float = None,
    ):
        self.state = None
        self.action = action
        self.parent = parent

        # Single version
        self.visit_count = 0  # N
        self.total_value = 0  # W
        self.prior = prior_prob  # P
        self.children = {}

        # Children version
        # self.children_visit = np.zeros(n_children)
        # self.children_total_value = np.zeros(n_children)
        # self.children_priors = np.zeros(n_children)
        # puct = child.value_sum / child.visit_count + child.prior * sqrt(parent.visit_count) / (child.visit_count + 1)

    def total_expansions(self):
        return int(self.expanded()) + sum(
            [c.total_expansions() for c in self.children.values()]
        )

    @property
    def mean_value(self) -> float:  # Q
        if self.visit_count > 0:
            return self.total_value / self.visit_count
        else:
            return 0

    def puct_score(self):
        puct_u = self.prior * np.sqrt(self.parent.visit_count) / (self.visit_count + 1)

        return self.mean_value + puct_u

    def select_action(self, temperature: float = 0.0):
        visit_counts = np.array([c.visit_count for c in self.children.values()])
        actions = np.array(list(self.children.keys()))

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == np.inf:
            action = np.random.choice(actions)
        else:
            probs = visit_counts ** (1 / temperature)
            probs = probs / probs.sum()
            action = np.random.choice(actions, p=probs)
        return action

    def select_child(self) -> Tuple[int, "Node"]:
        """Returns the child with greatest PUCT score and the action to take to get it

        Returns:
            Tuple[int, "Node"]: A tuple containing the `action` and the best child Node
        """
        puct_scores = [c.puct_score() for c in self.children.values()]
        best_action = list(self.children.keys())[np.argmax(puct_scores)]
        best_child = self.children[best_action]
        return best_action, best_child

    def expand(self, state, action_probs):
        self.state = state

        for action, prob in enumerate(action_probs):
            if not np.isnan(prob):
                self.children[action] = Node(
                    prior_prob=prob, action=action, parent=self
                )

    def expanded(self) -> bool:
        return len(self.children) > 0

    def update(self, value):
        self.total_value += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.update(value)


class MCTS:
    def __init__(
        self,
        env,
        model,
        dirichlet_noise: float = 0.03,
        dirichlet_epsilon: float = 0.25,
    ):
        # self.model = ModelServer(model=model, preprocessor=preprocessor, batch_size=16)
        # self.model = SimpleModel(model=model, preprocessor=preprocessor)
        self.env = env
        self.model = model

        self.dirichlet_noise = dirichlet_noise
        self.dirichlet_epsilon = dirichlet_epsilon

    def add_dirichlet_noise(self, action_probs: np.ndarray):
        action_noise = np.random.dirichlet(
            np.full(len(action_probs), self.dirichlet_noise)
        )

        action_probs = (
            action_probs * (1.0 - self.dirichlet_epsilon)
            + self.dirichlet_epsilon * action_noise
        )

        return action_probs

    def predict(self, state):
        action_probs, value = self.model.predict(state)

        return action_probs, value

    def expand_node(self, node: Node, state):
        action_probs, value = self.predict(state)
        action_probs = self.add_dirichlet_noise(action_probs)
        invalid_actions = ~self.env.get_valid_actions(state)
        action_probs[invalid_actions] = np.nan
        # action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.nansum(action_probs)
        node.expand(state, action_probs)

        return value

    def simulate(self, state, num_simulations: int = 1):
        # root = root or Node()
        root = Node()

        # Root node expansion
        if not root.expanded():
            self.expand_node(root, state)

        for _ in range(num_simulations):
            # print(sim)
            node = root

            done = False
            while not done:
                # Find the most promising leaf node
                while node.expanded():
                    action, node = node.select_child()

                # Do an step in the environment: get next state within an action and get the new state reward
                next_state, value, done = self.env.step(
                    state=node.parent.state, action=action
                )

                # If not terminal, expand
                if not done:
                    value = self.expand_node(node, next_state)

                # Update the tree
                node.update(value)
        return root
