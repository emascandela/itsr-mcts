from generators.generator import Generator


class ITSREnv:
    def __init__(self, *, generator: Generator):
        self.generator = generator

    @property
    def max_levels(self):
        return self.generator.max_transformation_sequence_length

    def reset(self):
        image_pair = self.generator.get_random_pair()
        return image_pair

    def step(self, state, action: int):
        next_state = self.get_next_state(state, action)
        reward, done = self.get_reward(next_state)

        return next_state, reward, done

    def get_action_size(self):
        return len(self.generator.transformations)

    def get_next_state(self, state, action):
        action_fn = self.generator.transformations[action]

        if not self.get_valid_actions(state)[action]:
            raise Exception(f"Error, {action_fn.__name__} is not allowed")

        state = state.apply_transformation(action_fn)
        return state

    def has_legal_actions(self, pair):
        return self.get_valid_actions(pair).any()

    def get_valid_actions(self, pair):
        actions, mask = self.generator.allowed_transformations(
            pair.source_image, return_mask=True
        )
        return mask

    def is_done(self, pair):
        return pair.issame()

    def get_reward(self, state):
        if self.is_done(state):
            return 1, True

        if self.has_legal_actions(state):
            return None, False

        return -1, True
