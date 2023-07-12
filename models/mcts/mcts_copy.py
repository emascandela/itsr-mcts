import math
from typing import List
import numpy as np
import time


class Chrono:
    def __init__(self, units: str = "s"):
        self.time_counts: List[float] = []
        self.started: float = None
        self.units = units
        self.get_time = time.perf_counter_ns if units == "ns" else time.perf_counter

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.started = self.get_time()

    def total_time(self):
        return sum(self.time_counts)

    def count(self):
        return len(self.time_counts)

    def mean_time(self):
        return self.total_time() / self.count()

    def stop(self):
        end = self.get_time()
        if self.started is None:
            raise Exception("Error, chrono not started")
        step_time_count = end - self.started
        self.time_counts.append(step_time_count)
        self.started = None
        return step_time_count

    def __add__(self, other: "Chrono"):
        new = Chrono()
        new.time_counts = self.time_counts + other.time_counts
        return new


def puct_score(parent, child, c_puct: float = 1.0):
    """
    The score for an action that would transition between the parent and child.
    """
    u = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    q = child.value()

    return q + c_puct * u


class Node:
    def __init__(self, prior, parent=None, prior_value=None, action=0, level=0):
        self.visit_count = 0
        self.level = level
        self.parent = parent
        self.prior = prior
        self.prior_value = prior_value
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.action = action

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array(
            [child.visit_count for child in self.children.values() if child is not None]
        )
        actions = [
            action
            for action in self.children.keys()
            if self.children[action] is not None
        ]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def children_probs(self):
        probs = [0 if c is None else c.prior for _, c in self.children.items()]
        return probs

    def expanded_children(self):
        return np.array(
            [False if c is None else c.expanded() for _, c in self.children.items()]
        )

    def select_child(self, c_puct: float = 1.0):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            if child is None:
                continue
            score = puct_score(self, child, c_puct)
            # print('ucb', score)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, action_probs=None, prior_value=None):
        # print(action_probs)
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.state = state
        self.prior_value = prior_value

        for a, prob in enumerate(action_probs):
            if prob != np.nan:
                self.children[a] = Node(
                    prior=prob, parent=self, action=a, level=self.level + 1
                )
            else:
                self.children[a] = None

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(
            self.state.__str__(), prior, self.visit_count, self.value()
        )

    def dot_repr(self, id):
        if self.state is None:
            return ""
        # formatted_state = [['x' if s_i == -1 else 'o' if s_i == 1 else '_' for s_i in s] for s in self.state]
        # ' '.join(formatted_state[0])
        # state = '\l'.join(['\|'+' '.join(s)+'\|' for s in formatted_state])
        state = ", ".join(self.state.applied_transformations)
        label = (
            f"{state}\l"
            f"\l"
            f"Name: {id}\l"
            f"Prior: {self.prior}\l"
            f"Visit Count: {self.visit_count}\l"
            f"Value: {self.value()}\l"
        )

        node_def = f'{id} [label="{label}"]'
        child_nodes = [
            self.children[c].dot_repr(id=id + f"_{c}")
            for c in self.children
            if self.children[c] is not None
        ]
        node_def = "\n".join([node_def, *child_nodes]) + "\n"
        if len(self.children) > 0:
            node_def += f"{id} -> " + ",".join(
                [id + f"_{i}" for i in self.children if self.children[i] is not None]
            )

        return node_def


class MCTS:
    def __init__(
        self,
        env,
        model,
        num_simulations,
        dirichlet_noise: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        c_puct: float = 1.0,
    ):

        self.env = env
        self.model = model
        self.num_simulations = num_simulations

        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_noise = dirichlet_noise
        self.c_puct = c_puct
        # self.args = args

    def run(
        self, model, state, root=None, initial_level=0, expand_first_level: bool = False
    ):

        chronos = {}

        def get_chrono(x) -> Chrono:
            if x not in chronos:
                chronos[x] = Chrono()

            return chronos[x]

        total_ch = get_chrono("total")
        total_ch.start()

        root = root or Node(0, level=initial_level)

        if root.level == 0 and expand_first_level:
            action_probs = (
                np.ones(self.env.get_action_size(), dtype=np.float32)
                / self.env.get_action_size()
            )
            action_probs[~self.env.get_valid_actions(state)] = np.nan
        # EXPAND root
        else:
            with get_chrono("predict"):
                action_probs, value = self.env.predict(self.model, state)
            action_probs = action_probs
        with get_chrono("expand"):
            root.expand(state, action_probs)

        info = {
            "expanded_nodes": 0,
            "expanded_nodes_per_level": np.zeros(
                self.env.max_levels + 1, dtype=np.int32
            ),
            "solved": False,
            "gt_transformations": state.target_image.applied_sequence_length,
        }
        info["expanded_nodes_per_level"][0] = 1

        i = 0

        # if expand_first_level:
        #     for action, c in root.children.items():
        #         next_state = self.env.get_next_state(root.state, action=action)
        #         value = self.env.get_reward(next_state)

        #         # if value is None:
        #         if value is None:
        #             action_probs, value = self.env.predict(model, next_state)
        #             c.expand(next_state, action_probs)
        #             info["expanded_nodes"] += 1
        #             info["expanded_nodes_per_level"][c.level - 1] += 1
        #             # if node.level in info["expanded_nodes_per_level"]:
        #             #     info["expanded_nodes_per_level"][node.level] += 1
        #             # else:
        #             #     info["expanded_nodes_per_level"][node.level] = 1
        #         elif value:
        #             info["solved"] = True
        #             c.state = next_state

        #         self.backpropagate([root, c], value)

        # for _ in range(self.num_simulations):
        while i < self.num_simulations:
            node = root
            search_path = [node]

            # SELECT
            with get_chrono("select"):
                while node.expanded():
                    action, node = node.select_child(self.c_puct)
                    search_path.append(node)

            # parent = search_path[-2]
            parent = node.parent
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            # next_state, _ = self.game.get_next_state(state, action=action)
            # prev_allowed = self.env.get_valid_actions(state)
            with get_chrono("next_state"):
                next_state = self.env.get_next_state(
                    state, action=action, parent=parent
                )
            # post_allowed = self.env.get_valid_actions(state)
            # Get the board from the perspective of the other player

            #     print(
            #         "TForms",
            #         ", ".join([t.__name__ for t in state.transformation_sequence]),
            #     )
            #     print("Initial level", initial_level)
            #     print("type:", type(state.source_image))
            #     print("Source figures", len(state.source_image.figures))
            #     print("Target figures", len(state.target_image.figures))
            #     print("Grid", state.source_image.grid)
            #     print("Inserts:", len(state.source_image.added_figures))
            #     print("Removes:", len(state.source_image.removed_figures))
            #     print(
            #         "Allowed actions",
            #         len(self.env.generator.allowed_transformations(state.source_image)),
            #     )
            #     print("ENV Allowed actions", self.env.get_valid_actions(state))
            #     quit()
            with get_chrono("reward"):
                value = self.env.get_reward(next_state)
            if value is None:
                with get_chrono("predict"):
                    action_probs, value = self.env.predict(model, next_state)

                if node.level == 0:
                    action_probs = np.ones_like(action_probs) / len(action_probs)
                    print("EEEP! Node is 0")
                    pass
                else:
                    action_probs = action_probs * (
                        1 - self.dirichlet_epsilon
                    ) + self.dirichlet_epsilon * np.random.dirichlet(
                        np.full(len(action_probs), self.dirichlet_noise)
                    )
                with get_chrono("expand"):
                    node.expand(next_state, action_probs)
                info["expanded_nodes"] += 1
                # info["expanded_nodes_per_level"][node.level] += 1
                # if node.level in info["expanded_nodes_per_level"]:
                #     info["expanded_nodes_per_level"][node.level] += 1
                # else:
                #     info["expanded_nodes_per_level"][node.level] = 1
            else:
                i += 1
                node.state = next_state
                if value:
                    info["solved"] = True

            with get_chrono("backprop"):
                self.backpropagate(search_path, value)

        total_ch.stop()
        info["chronos"] = chronos
        return root, info

    def backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1

    def dot_repr(self, root):
        repr = root.dot_repr("node0")
        dot_data = "digraph L { " "node [shape=record fontname=Consolas];" f"{repr}" "}"
        return dot_data

    def save_graph_fig(self, root, path):
        import graphviz

        graph = graphviz.Source(self.dot_repr(root), format="pdf")
        graph.render(path)


class EvalMCTS:
    def __init__(self, env, model, node_selection: str = "best"):

        self.env = env
        self.model = model
        self.node_selection = node_selection

    def run(self, model, state):
        max_expansions = 100
        root = Node(0)

        # EXPAND root
        action_probs, value = self.env.predict(self.model, state)

        root.expand(state, action_probs, prior_value=value)

        nodes = [root]

        info = {
            "expanded_nodes": 0,
            "expanded_nodes_per_level": np.zeros(
                self.env.max_levels + 1, dtype=np.int32
            ),
            "pred_sequence": [],
            "solved": False,
            "gt_transformations": state.target_image.applied_sequence_length,
            # "leaf_reached": 0,
        }

        while True:
            if len(nodes) == 0:
                break
            if self.node_selection == "best":
                best = np.argmax([n.prior_value for n in nodes])
                node = nodes[best]
            elif self.node_selection == "last":
                node = nodes[-1]
                nodes = []
            else:
                Exception()

            action_probs = np.array(node.children_probs())
            value = node.prior_value

            valid_moves = (
                self.env.get_valid_actions(node.state) & ~node.expanded_children()
            )
            action_probs[~valid_moves] = np.nan
            action = np.nanargmax(action_probs)
            info["pred_sequence"].append(action)

            child_to_expand = node.children[action]
            info["expanded_nodes"] += 1
            info["expanded_nodes_per_level"][node.level] += 1

            new_state = self.env.get_next_state(node.state, action=action)
            rew = self.env.get_reward(new_state)

            if rew is None:
                # Expand
                action_probs, value = self.env.predict(model, new_state)
                child_to_expand.expand(new_state, action_probs, prior_value=value)
                nodes.append(child_to_expand)
            elif rew == 1:
                info["solved"] = True
                break
            else:
                child_to_expand.expand(
                    new_state, action_probs=np.zeros(len(node.children))
                )

            if info["expanded_nodes"] >= max_expansions:
                break
            valid_moves = (
                self.env.get_valid_actions(node.state) & ~node.expanded_children()
            )
            if valid_moves.sum() == 0:
                if node in nodes:
                    nodes.remove(node)

        return root, info

    def backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
