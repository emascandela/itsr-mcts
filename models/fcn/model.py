from models.mcts.itsr_env import ITSREnv
import time
from ..model import Model
import torch
from dataclasses import dataclass, field
from typing import Dict, Any
import progressbar
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from models.serving import ModelServer
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm


class Node:
    def __init__(self, value: int = 0, action=None, parent=None, state=None):
        self.value = value
        self.parent = parent
        self.action = action
        self.state = state
        self.children = {}

    def expand(self, child_priors: np.ndarray, state):
        self.state = state

        for i, p in enumerate(child_priors):
            if not np.isnan(p):
                self.children[i] = Node(value=self.value * p, parent=self, action=i)
        return self.children


class EvalTree:
    def __init__(self, env, model, max_expansions: int = 100):

        self.env = env
        self.model = model
        self.max_expansions = max_expansions

    def predict(self, state):
        action_probs, *_ = self.model.predict(state)

        invalid_actions = ~self.env.get_valid_actions(state)
        action_probs[invalid_actions] = np.nan
        # action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.nansum(action_probs)

        return action_probs

    def simulate(self, state):
        root = Node(1, state=state)

        # EXPAND root
        action_probs = self.predict(state)
        children = root.expand(child_priors=action_probs, state=state)

        nodes = list(children.values())

        expanded_nodes = 0

        while True:
            if len(nodes) == 0:
                break

            # Find node with best value
            best = np.argmax([n.value for n in nodes])
            node = nodes.pop(best)

            state, reward, done = self.env.step(
                state=node.parent.state, action=node.action
            )
            expanded_nodes += 1

            if done:
                if reward == 1:
                    return expanded_nodes
            else:
                action_probs = self.predict(state)
                children = node.expand(child_priors=action_probs, state=state)
                nodes += list(children.values())

            if expanded_nodes > self.max_expansions:
                break

        return np.inf


@dataclass
class Config:
    batch_size: int
    steps_per_epoch: int
    validation_steps: int
    evaluation_steps: int = 1000
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {})
    epochs: int = None
    evaluation_threads: int = 128
    evaluation_batch_size: int = 64
    generator_params: Dict[str, Any] = field(default_factory=lambda: {})
    rollout_threads: int = None
    rollout_batch_size: int = None


class FCNModel(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)

        self.train_generator = self.generator_class(**self.train_generator_params)
        self.val_generator = self.generator_class(**self.val_generator_params)
        self.test_generator = self.generator_class(**self.test_generator_params)

        self.general_config = Config(**self.train_config)

        self.test_env = ITSREnv(
            generator=self.test_generator,
        )

        self.val_env = ITSREnv(
            generator=self.val_generator,
        )

        self.model = self.build_model()

        self.serving_model = ModelServer(
            self.model,
            preprocessor=self.preprocessor,
            batch_size=self.general_config.evaluation_batch_size,
            # timeout_ms=100,
        )

    def train(self):
        self.model.train()

        optimizer_params = self.general_config.optimizer_params
        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [12, 20], gamma=0.1)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_scaler = torch.cuda.amp.GradScaler()

        train_dataloader = self.preprocessor.get_dataset(
            self.train_generator,
            batch_size=self.general_config.batch_size,
            n_samples=self.general_config.steps_per_epoch,
        )
        val_dataloader = self.preprocessor.get_dataset(
            self.val_generator,
            # steps=self.general_config.steps_per_epoch,
            batch_size=self.general_config.batch_size,
            n_samples=self.general_config.validation_steps,
        )

        start_time = time.time()

        for epoch in range(self.general_config.epochs):
            epoch_loss = 0.0
            epoch_acc = 0
            st = time.perf_counter()
            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                for inp in inputs:
                    inputs[inp] = inputs[inp].cuda()
                    # inputs[inp] = inputs[inp]

                optimizer.zero_grad()
# 
                with torch.cuda.amp.autocast(enabled=False):
                    outputs = self.model(**inputs)
                    loss = loss_fn(outputs, labels.cuda())
                loss_scaler.scale(loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()

                output_labels = torch.argmax(outputs, axis=1).cpu().detach()
                epoch_acc += torch.mean((output_labels == labels).to(torch.float32))

                epoch_loss += loss

                if (i + 1) % 100 == 0:
                    elapsed_time = time.time() - start_time
                    total_steps = (
                        self.general_config.steps_per_epoch
                        / self.general_config.batch_size
                        * self.general_config.epochs
                    )
                    steps_done = (
                        self.general_config.steps_per_epoch
                        / self.general_config.batch_size
                        * epoch
                        + i
                    )
                    estimated_time = (elapsed_time / steps_done) * (
                        total_steps - steps_done
                    )
                    time_fmt = "%H:%M:%S"
                    elapsed = time.strftime(time_fmt, time.gmtime(elapsed_time))
                    estimated = time.strftime(time_fmt, time.gmtime(estimated_time))
                    print(
                        f"Epoch {epoch+1}/{self.general_config.epochs} {i+1} elapsed: {elapsed} eta: {estimated} loss: {epoch_loss / i :.4f} acc: {epoch_acc / i :.4f}"
                    )
                st = time.perf_counter()

            val_loss = []
            val_acc = []
            for data in val_dataloader:
                inputs, labels = data
                for inp in inputs:
                    inputs[inp] = inputs[inp].cuda()

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    loss = loss_fn(outputs, labels.cuda())
                output_labels = torch.argmax(outputs, axis=1).cpu().detach()
                val_acc.append((output_labels == labels).to(torch.float32))
                val_loss.append(loss.cpu().detach())

            val_loss = torch.mean(torch.stack(val_loss, dim=0))
            val_acc = torch.mean(torch.cat(val_acc, dim=0))
            print(
                f"Epoch {epoch+1}/{self.general_config.epochs} loss: {val_loss:.4f} acc: {val_acc:.4f}"
            )

            scheduler.step()

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

            state, reward, done = self.env.step(state, action=action)
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
            # timeout_ms=100,
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
