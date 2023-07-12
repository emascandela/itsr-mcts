import numpy as np
import torch
import concurrent
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import math
from tqdm import tqdm
from random import shuffle
import multiprocessing
from models.fcn.model import EvalTree
import gc
from models.serving import ModelServer as ModelServer

# from models import preprocess
from models.mcts.itsr_env import ITSREnv

# from models import preprocess
# import tensorflow as tf


from models.mcts.mcts import MCTS

# from models.mcts.mcts import MCTS, EvalMCTS, Chrono
from ..model import Model
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class Config:
    batch_size: int
    steps_per_epoch: int
    rollout_batch_size: int = 64
    rollout_threads: int = 128
    evaluation_threads: int = 128
    evaluation_batch_size: int = 32
    dirichlet_noise: float = 0.03
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.1
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {})
    num_simulations: int = None
    num_episodes: int = None
    epochs: int = None
    generator_params: Dict[str, Any] = field(default_factory=lambda: {})
    replay_size: int = None
    c_puct: float = 1.0
    evaluation_steps: int = 1000


class MCTSModel(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)

        self.val_generator = self.generator_class(**self.val_generator_params)
        self.test_generator = self.generator_class(**self.test_generator_params)

        if "curriculum" in self.train_config:
            self.curriculum_config = [
                Config(**{**self.train_config["general"], **conf})
                for conf in self.train_config["curriculum"]
            ]
        else:
            self.curriculum_config = [self.general_config]

        self.general_config = Config(**self.train_config["general"])
        # self.summary_writer = tf.summary.create_file_writer(self.get_tensorboard_path())

        self.val_env = ITSREnv(
            generator=self.val_generator,
        )
        self.test_env = ITSREnv(
            generator=self.test_generator,
        )

        self.input_shape = (
            self.val_generator.image_size,
            self.val_generator.image_size,
            3,
        )

        self.replay_buffer = []

        self.model = self.build_model()

    def train_curriculum_iter(
        self, config, prev_epochs: int = 0, total_epochs: int = None
    ):
        total_epochs = total_epochs or (config.epochs + prev_epochs)

        optimizer = torch.optim.SGD(self.model.parameters(), **config.optimizer_params)
        policy_loss_fn = torch.nn.CrossEntropyLoss()
        value_loss_fn = torch.nn.MSELoss()
        loss_scaler = torch.cuda.amp.GradScaler()

        # self.model.compile(
        #     loss={
        #         "policy": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        #         "value": tf.keras.losses.MeanSquaredError(),
        #     },
        #     optimizer=tf.keras.optimizers.SGD(**config.optimizer_params),
        # )
        self.serving_model = ModelServer(
            self.model,
            preprocessor=self.preprocessor,
            batch_size=config.rollout_batch_size,
            timeout_ms=0,
        )

        val_generator = self.generator_class(**self.val_generator_params)

        for epoch in range(prev_epochs, prev_epochs + config.epochs):
            print("")
            print(f"Epoch {epoch+1}/{total_epochs}")

            self.serving_model.start()
            training_samples = self.get_training_samples(config=config)

            # for t in training_samples:
            #     pair, policy, value = t
            #     print(
            #         [
            #             val_generator.transformations.index(t)
            #             for t in pair.target_image.applied_transformations
            #         ],
            #         policy,
            #     )

            self.serving_model.stop()
            rewards = np.array([s[-1] for s in training_samples])
            probs = np.array([s[-2] for s in training_samples])
            print(f"Reward mean: {rewards.mean():.3f}")
            print(f"Probs mean: {probs.mean(0)}")

            self.replay_buffer.extend(training_samples)

            if config.replay_size is not None:
                self.replay_buffer = self.replay_buffer[-config.replay_size :]

            train_dataloader = self.preprocessor.get_dataset(
                val_generator,
                data=self.replay_buffer,
                batch_size=config.batch_size,
            )

            self.fit(
                train_dataloader,
                config,
                epoch=epoch,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                policy_loss_fn=policy_loss_fn,
                value_loss_fn=value_loss_fn,
            )

            if config.replay_size is None:
                self.train_examples = []

            if (epoch > 10) and (epoch % 1 == 0):
                print("Running validation")
                single_shot_accuracy = self.evaluate_single_shot(env=self.val_env)
                print(f"Single-hot Accuracy: {single_shot_accuracy:.4f}")

                k_retrievals = self.evaluate_top_k(env=self.val_env)
                k_values = list(range(10, 101, 15))
                print("TopK Accuracy: ", end="")
                print(
                    *[f"K={k}: {(k_retrievals <= k).mean():.4f}" for k in k_values],
                    sep=" ",
                )

    def train(self):
        epoch = 0
        if self.curriculum_config is not None:
            total_epochs = sum([c.epochs for c in self.curriculum_config])
            for c_iter, c_config in enumerate(self.curriculum_config):
                print(f"Running curriculum iteration {c_iter}")
                self.train_curriculum_iter(
                    c_config, prev_epochs=epoch, total_epochs=total_epochs
                )
                self.save_weights(name=f"checkpoint_{c_iter}")
                epoch += c_config.epochs
        else:
            self.train_curriculum_iter(c_config)

    def get_training_samples(self, config: Config):
        generator = self.generator_class(
            **{**self.generator_params, **config.generator_params}
        )
        env = ITSREnv(
            generator=generator,
        )

        # generator = self.generator_class(**config.generator_params)
        # val_generator = self.generator_class(**self.val_generator_params)

        self.model.eval()

        with ThreadPoolExecutor(config.rollout_threads) as executor:
            # with ProcessPoolExecutor(8) as executor:
            futures = [
                executor.submit(self.execute_episode, env, config)
                for _ in range(config.num_episodes)
            ]

            results = []
            for fut in tqdm(as_completed(futures), total=len(futures)):
                results.extend(fut.result())

        return results

    def execute_episode(self, env, config: Config = None):
        config = config or self.general_config

        train_examples = []

        mcts = MCTS(
            env=env,
            model=self.serving_model,
            dirichlet_epsilon=config.dirichlet_epsilon,
            dirichlet_noise=config.dirichlet_noise,
        )
        state = env.reset()

        done = False
        while not done:
            node = mcts.simulate(state=state, num_simulations=config.num_simulations)
            # print(node.total_expansions())
            action_probs = np.zeros(env.get_action_size())
            for k, v in node.children.items():
                action_probs[k] = v.visit_count
            action_probs[~env.get_valid_actions(state)] = 0
            action_probs /= np.sum(action_probs)

            train_examples.append((state, action_probs))

            action = node.select_action(temperature=config.temperature)
            # print(action, env.get_valid_actions(state), state.transformation_sequence)
            node = node.children[action]

            state, reward, done = env.step(state, action)
            # print("done", done, "r", reward, "same", state.issame(), action_probs)

        return [(*s, reward) for s in train_examples]

    def predict(self, state, env: ITSREnv = None):
        env = env if env is not None else self.test_env
        done = False
        pred_sequence = []
        while not done:
            action_probs, *_ = self.serving_model.predict(state)
            invalid_actions = ~env.get_valid_actions(state)
            action_probs[invalid_actions] = np.nan
            # action_probs = action_probs * valid_moves  # mask invalid moves
            action_probs /= np.nansum(action_probs)

            action = np.nanargmax(action_probs)
            pred_sequence.append(action)

            state, reward, done = env.step(state, action=action)
        return action, reward

    def evaluate_single_shot(self, env: ITSREnv = None):
        env = env if env is not None else self.test_env

        self.model.eval()
        self.serving_model.start()

        with ThreadPoolExecutor(
            max_workers=self.general_config.evaluation_threads
        ) as executor:
            futures = [
                executor.submit(self.predict, env.reset(), env)
                for _ in range(self.general_config.evaluation_steps)
            ]

            rewards = []
            for fut in tqdm(as_completed(futures), total=len(futures)):
                _, reward = fut.result()
                rewards.append(reward)

        self.serving_model.stop()
        rewards = np.array(rewards)
        solves = rewards == 1
        return solves.mean()

    def evaluate_top_k(self, env: ITSREnv = None):
        env = env if env is not None else self.test_env

        self.model.eval()
        self.serving_model = ModelServer(
            self.model,
            preprocessor=self.preprocessor,
            batch_size=self.general_config.evaluation_batch_size,
            timeout_ms=100,
        )
        self.serving_model.start()

        tree = EvalTree(env, model=self.serving_model)
        with ThreadPoolExecutor(
            max_workers=self.general_config.evaluation_threads
        ) as executor:
            futures = [
                executor.submit(tree.simulate, env.reset())
                for _ in range(self.general_config.evaluation_steps)
            ]

            results = []
            for fut in tqdm(as_completed(futures), total=len(futures)):
                results.append(fut.result())

        self.serving_model.stop()
        return np.array(results)

    # def predict_single_shot(self, state=None, env: ITSREnv = None):
    #     if state is None:
    #         state = env.reset()

    #     done = False
    #     while not done:
    #         action_probs, _ = self.serving_model.predict(state)
    #         action = torch.argmax(action_probs).cpu().detach().numpy()
    #         state, reward, done = self.env.step(state=state, action=action)
    #     return reward

    def fit(
        self,
        dataloader,
        config: Config,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        loss_scaler,
        policy_loss_fn,
        value_loss_fn,
    ):
        # print("fitting", self.model.device())
        # self.model.train()
        import time

        i = 1
        for step in range(1, config.steps_per_epoch + 1):
            curr_loss = 0

            for i, data in enumerate(dataloader, start=i):

                # inputs, outputs = data
                # policy = outputs["policy"]
                # value = outputs["value"]
                # print(policy.shape)
                # for inp in inputs:
                #     inputs[inp] = inputs[inp].cuda()
                source, target, policy, value = data

                optimizer.zero_grad()

                # pred_policy, pred_value = self.model(
                #     source_image=source.cuda(), target_image=target.cuda()
                # )
                # # pred_policy, pred_value = self.model(
                # #     source_image=source, target_image=target
                # # )
                # policy_loss = policy_loss_fn(pred_policy, policy.cuda())
                # value_loss = value_loss_fn(pred_value, value.cuda())
                # # policy_loss = policy_loss_fn(pred_policy, policy)
                # # value_loss = value_loss_fn(pred_value, value)
                # loss = policy_loss + value_loss
                # loss.backward()
                # optimizer.step()

                # loss_scaler.scale(loss).backward()
                # loss_scaler.step(optimizer)
                # loss_scaler.update()

                with torch.cuda.amp.autocast(enabled=True):
                    pred_policy, pred_value = self.model( source_image=source.cuda(), target_image=target.cuda())
                    policy_loss = policy_loss_fn(
                        pred_policy, policy.to(torch.float32).cuda()
                    )
                    value_loss = value_loss_fn(
                        pred_value, value.to(torch.float32).cuda()
                    )
                    loss = policy_loss + value_loss
                curr_loss += loss
                loss_scaler.scale(loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()

            print(
                f"Epoch {epoch+1} train iter {step} loss {curr_loss / len(dataloader):.4f}"
            )

        # history = self.model.fit(dataset, steps_per_epoch=2, verbose=1)
