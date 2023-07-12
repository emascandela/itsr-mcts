import numpy as np
from random import shuffle
from models.fcn.model import EvalTree
import gc

# from models import preprocess
from models.mcts.itsr_env import ITSREnv

# from models import preprocess
import progressbar
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

# import torch
# import torch.optim as optim
import tensorflow as tf
from ..common import backbones


from models.mcts.mcts import MCTS, EvalMCTS, Chrono
from ..model import Model
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class CurriculumStepConfig:
    num_simulations: int = None
    num_episodes: int = None
    epochs: int = None
    steps_per_epoch: int = None
    temperature: float = None
    generator_params: Dict[str, Any] = field(default_factory=lambda: {})
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {})
    replay_size: int = None


@dataclass
class Config:
    batch_size: int
    steps_per_epoch: int
    rollout_batch_size: int = 64
    rollout_threads: int = 128
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

        self.general_config = Config(**self.train_config["general"])
        if "curriculum" in self.train_config:
            self.curriculum_config = [
                Config(**(conf | self.train_config["general"]))
                for conf in self.train_config["curriculum"]
            ]
        else:
            self.curriculum_config = [self.general_config]

        self.summary_writer = tf.summary.create_file_writer(self.get_tensorboard_path())

        self.val_env = ITSREnv(
            generator=self.val_generator,
            preprocessor=self.preprocessor,
            batch_size=self.general_config.rollout_batch_size,
        )
        self.test_env = ITSREnv(
            generator=self.test_generator,
            preprocessor=self.preprocessor,
            batch_size=self.general_config.rollout_batch_size,
        )

        shape = (self.val_generator.image_size, self.val_generator.image_size, 3)

        self.model = self.arch.build(
            input_shape=shape, n_classes=len(self.val_generator.transformations)
        )

        self.train_examples = []

    def execute_episode(self, env, config, state=None):
        chronos = {}
        chronos["execution_total"] = Chrono()
        chronos["execution_total"].start()

        train_examples = []

        if state is None:
            state = env.reset()
        initial_level = 0
        infos = []
        node = None

        num_simulations = config.num_simulations or self.general_config.num_simulations
        mcts = MCTS(
            env,
            self.model,
            num_simulations,
            dirichlet_epsilon=self.general_config.dirichlet_epsilon,
            dirichlet_noise=self.general_config.dirichlet_noise,
            c_puct=self.general_config.c_puct,
        )
        while True:
            if node is not None:
                mcts.num_simulations = max(num_simulations - node.visit_count, 1)
            node, info = mcts.run(
                self.model,
                state.copy(),
                root=node,
                initial_level=initial_level,
            )
            for k, ch in info["chronos"].items():
                if k in chronos:
                    chronos[k] += ch
                else:
                    chronos[k] = ch
            infos.append(info)

            action_probs = [0 for _ in range(env.get_action_size())]
            for k, v in node.children.items():
                if v is None:
                    continue
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)

            train_examples.append(
                (
                    state.preprocessing_sequence,
                    state.gt_sequence,
                    state,
                    action_probs,
                )
            )

            action = node.select_action(
                temperature=config.temperature or self.general_config.temperature
            )
            # parent = node
            node = node.children[action]
            state = env.get_next_state(state, action)
            initial_level += 1
            reward = env.get_reward(state)
            # print(reward)

            if reward is not None:
                ret = []
                for sample in train_examples:
                    ret.append(
                        [
                            *sample,
                            reward,
                        ]
                    )

                info = {
                    "expanded_nodes": np.sum([i["expanded_nodes"] for i in infos]),
                    # "expanded_nodes_per_level": np.sum(
                    #     [n["expanded_nodes_per_level"] for i, n in enumerate(infos)],
                    #     axis=0,
                    # ),
                    "expanded_nodes_per_level": {
                        i: inf["expanded_nodes"] for i, inf in enumerate(infos)
                    },
                    "solved": np.any([i["solved"] for i in infos]),
                    "gt_transformations": node.state.target_image.applied_sequence_length,
                }

                chronos["execution_total"].stop()
                info["chronos"] = chronos

                del node
                gc.collect()

                return ret, info

    def train(self, starting_iter: int = 0):
        epochs = 0
        with self.summary_writer.as_default():
            self.logger.info(f"Starting training on model {self.name}")
            if self.curriculum_config:
                for curriculum_iter, config in enumerate(self.curriculum_config):
                    step_epochs = config.epochs or self.general_config.epochs or 0

                    if curriculum_iter < starting_iter:
                        self.logger.info(
                            f"Skipping curriculum iteration {curriculum_iter}"
                        )
                        epochs += step_epochs
                        continue
                    self.logger.info(f"Running curriculum iteration {curriculum_iter}")
                    self.train_curriculum_iter(config, epochs)
                    self.save_weights(name=f"checkpoint_{curriculum_iter}")

                    epochs += step_epochs
            else:
                self.train_curriculum_iter(self.general_config)

    def train_curriculum_iter(self, config, prev_epochs=0):
        optimizer_params = {
            **self.general_config.optimizer_params,
            **config.optimizer_params,
        }
        self.model.compile(
            loss={
                "policy": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                "value": tf.keras.losses.MeanSquaredError(),
            },
            optimizer=tf.keras.optimizers.SGD(**optimizer_params),
        )

        generator = self.generator_class(
            **{**self.generator_params, **config.generator_params}
        )
        val_generator = self.generator_class(**self.val_generator_params)

        env = ITSREnv(
            generator=generator,
            preprocessor=self.preprocessor,
            batch_size=self.general_config.rollout_batch_size,
        )
        val_env = ITSREnv(
            generator=val_generator,
            preprocessor=self.preprocessor,
            batch_size=self.general_config.rollout_batch_size,
        )
        for i in range(
            prev_epochs,
            prev_epochs + (config.epochs or self.general_config.epochs or 0),
        ):
            self.logger.info(
                f"Epoch {i+1}/{config.epochs or self.general_config.epochs}"
            )
            tf.summary.scalar(
                "max_transformations", generator.max_transformations, step=i
            )
            tf.summary.scalar("max_repetitions", generator.max_transformations, step=i)

            if (i) % 5 == 0 and i != 0:
                self.logger.info("Performing validation")

                print("Single shot")
                # result = self.evaluate(10000, use_mcts=False)
                result = self.evaluate_single_shot(env=val_env)
                # result = self.evaluate(100, use_mcts=False)
                solved = np.array(result["solved"])
                # n = np.array([r["gt_transformations"] for r in result])
                n = np.array(result["gt_transformations"])

                self.logger.info(
                    f"Single shot evaluation: [Mean solved={solved.mean():.2f}]"
                )
                tf.summary.scalar(f"single_shot_acc", solved.mean(), step=i)
                result_str = []
                for n_i in np.sort(np.unique(n)):
                    mask = n == n_i
                    result_str.append(f"{n_i}: acc={np.mean(solved[mask]*100):.1f}%")
                tf.summary.scalar(
                    f"single_shot_acc_per_len", np.mean(solved[mask]), step=i
                )
                self.logger.info(
                    f"Single shot evaluation (per length): [{' | '.join(result_str)}]"
                )

                print("")

                print("TopK")
                # result = self.evaluate(1000, True)
                result = self.evaluate(env=val_env)
                solved = np.array(result["solved"])
                # print(f"Mean solved: {solved.mean():.2f}")
                tf.summary.scalar("credits_acc", solved.mean(), step=i)

                self.logger.info(
                    f"K-credits evaluation: [Mean solved={solved.mean():.2f}]"
                )
                # solved = np.array([r["solved"] for r in result])
                # expanded_nodes = np.array([r["expanded_nodes"] for r in result])
                expanded_nodes = np.array(result["expanded_nodes"])
                # n = np.array([r["gt_transformations"] for r in result])
                n = np.array(result["gt_transformations"])
                result_str = []

                for n_i in np.sort(np.unique(n)):
                    mask = n == n_i

                    total_solves = np.mean(solved[mask])
                    mean_steps = np.mean(expanded_nodes[mask][solved[mask]])
                    result_str.append(
                        f"{n_i}: acc={total_solves*100:.1f}% steps={mean_steps:.1f}"
                    )

                    tf.summary.scalar(f"credits_acc_len={n_i}", total_solves, step=i)
                # print(" | ".join(result_str))
                self.logger.info(
                    f"K-credits evaluation (per length): [{' | '.join(result_str)}]"
                )

                print("\n")
                # print("=" * 80)
                # print("=" * 80)

                # print("")
                # print("")
            # train_examples = []
            results = []
            infos = []

            self.logger.info("Rolling out")
            # print("model", env.model_mutex.locked())
            # print("model_wait", env.model_wait_mutex.locked())
            # print("data", env.data_mutex.locked())
            # print("data_wait", env.data_wait_mutex.locked())
            # print("process_finished", env.process_finished_mutex.locked())
            gc.collect()

            with progressbar.ProgressBar(
                max_value=config.num_episodes or self.general_config.num_episodes
            ) as pb:
                with ThreadPoolExecutor(
                    self.general_config.rollout_threads
                ) as executor:

                    def run_single_episode(state=None):
                        res = self.execute_episode(env=env, config=config, state=state)
                        pb.update(pb.value + 1)

                        return res

                    for eps in range(
                        config.num_episodes or self.general_config.num_episodes
                    ):
                        # iteration_train_examples = self.execute_episode()
                        # train_examples.extend(iteration_train_examples)
                        results.append(executor.submit(run_single_episode))
            rewards = []
            for res in results:
                # try:
                sample, info_i = res.result()
                for s in sample:
                    rewards.append(s[-1])
                self.train_examples.extend(sample)
                infos.append(info_i)

            # with progressbar.ProgressBar(
            #     max_value=config.num_episodes or self.general_config.num_episodes
            # ) as pb:

            #     def run_single_episode():
            #         res = self.execute_episode(env=env, config=config)
            #         pb.update(pb.value + 1)

            #         return res

            #     for eps in range(
            #         config.num_episodes or self.general_config.num_episodes
            #     ):
            #         # iteration_train_examples = self.execute_episode()
            #         # train_examples.extend(iteration_train_examples)
            #         results.append(run_single_episode())
            # for res in results:
            #     # try:
            #     sample, info_i = res
            #     self.train_examples.extend(sample)
            #     infos.append(info_i)

            #     # except:
            #     #     pass
            # rewards = np.array([s[-1] for s in self.train_examples])
            expanded_nodes = np.array([info["expanded_nodes"] for info in infos])

            chronos = {}
            for info in infos:
                for k, ch in info["chronos"].items():
                    if k in chronos:
                        chronos[k] += ch
                    else:
                        chronos[k] = ch
            self.logger.info(
                f"Mean reward: {np.mean(rewards):.2} Expanded nodes: {np.mean(expanded_nodes)}"
            )
            tf.summary.scalar("reward", np.mean(rewards), step=i)
            tf.summary.scalar("expanded_nodes", np.mean(expanded_nodes), step=i)
            # for k in chronos:
            #     print(
            #         f"{k}: {chronos[k].total_time():.4f}s {chronos[k].mean_time():4}s"
            #     )

            if False:
                solved = np.array([i["solved"] for i in infos])
                expanded_nodes = np.array([i["expanded_nodes"] for i in infos])
                expanded_nodes_per_level = np.array(
                    [i["expanded_nodes_per_level"] for i in infos]
                )
                N = np.array([i["gt_transformations"] for i in infos])

                for n_i in np.sort(np.unique(N)):
                    mask = N == n_i

                    total_solves = np.mean(solved[mask])
                    mean_steps = np.mean(expanded_nodes[mask][solved[mask]])

                    mean_expanded_nodes_per_level = {}
                    for en in expanded_nodes_per_level[mask]:
                        for i, n in en.items():
                            mean_expanded_nodes_per_level[i] = (
                                mean_expanded_nodes_per_level.get(i) or []
                            ) + [n]
                    for i in mean_expanded_nodes_per_level.keys():
                        mean_expanded_nodes_per_level[i] = np.mean(
                            mean_expanded_nodes_per_level[i]
                        )

                    self.logger.info(
                        f"{n_i}: Acc: {total_solves:.4f} Steps: {mean_steps:.4f} Nodes/level: {', '.join(f'{i}: {n:.2f}' for i, n in mean_expanded_nodes_per_level.items())}"
                    )

            # shuffle(train_examples)
            # quit()
            replay_size = config.replay_size or self.general_config.replay_size
            if replay_size is not None:
                self.train_examples = self.train_examples[-replay_size:]
            # self.train_examples = self.train_examples[-20000:]
            self.train_samples(generator, self.train_examples, config, step=i)
            if replay_size is None:
                self.train_examples = []

            self.save_weights()
            tf.summary.flush()
            # self.save_weights()

    @tf.function
    def train_step(self, state, target_pis, target_vs):
        with tf.GradientTape() as tape:
            out_pi, out_v = self.model(state, training=True)
            # tf.print(target_pis)
            pi_loss = self.loss_pi(target_pis, out_pi)
            v_loss = self.loss_v(target_vs, out_v)
            total_loss = pi_loss + v_loss
            scaled_loss = self.model.optimizer.get_scaled_loss(total_loss)

        scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
        grads = self.model.optimizer.get_unscaled_gradients(scaled_grads)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return pi_loss, v_loss

    def train_samples(self, generator, examples, train_config, step):
        batch_size = self.general_config.batch_size

        dataset = self.preprocessor.get_dataset(
            generator,
            examples,
            batch_size,
        )

        steps_per_epoch = (
            train_config.steps_per_epoch or self.general_config.steps_per_epoch
        )
        total_steps = len(self.train_examples) * steps_per_epoch

        # dataset = dataset.take(2000 // batch_size)
        # dataset = dataset.cache()
        # dataset = dataset.repeat(steps_per_epoch)
        for d in dataset:
            print(d)

        history = self.model.fit(dataset, steps_per_epoch=(total_steps // batch_size))
        for k, v in history.history.items():
            self.logger.info(f"{k}: mean={np.mean(v):.3f}")
            tf.summary.scalar(k, np.mean(v), step=step)
        return

        # total_steps = (
        #     self.general_config.steps_per_epoch
        #     // self.general_config.batch_size
        #     # int(len(examples) / self.general_config.batch_size)
        #     # * self.general_config.steps_per_epoch
        # )
        progbar = tf.keras.utils.Progbar(
            total_steps // batch_size,
            verbose=1,
            # stateful_metrics=["pi_loss", "v_loss"],
        )

        # for step in range(total_steps):
        for ds_i in dataset.take(total_steps // batch_size):
            # sample_ids = np.random.randint(
            #     len(examples), size=self.general_config.batch_size
            # )

            # prep_tf, gt_tf, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

            # # state_source = np.array(state_source, dtype=np.float32)
            # # state_target = np.array(state_target, dtype=np.float32)

            # target_pis = np.array(pis).astype(np.float32)
            # target_vs = np.array(vs).astype(np.float32)

            # compute output
            # with tf.GradientTape() as tape:
            #     out_pi, out_v = self.model((preprocess(state_source), preprocess(state_target)), training=True)
            #     pi_loss = self.loss_pi(target_pis, out_pi)
            #     v_loss = self.loss_v(target_vs, out_v)
            #     total_loss = pi_loss + v_loss

            # grads = tape.gradient(total_loss, self.model.trainable_variables)
            # optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            # preprocessed_state = self.preprocessor.preprocess_batch(state)
            # state_source = preprocessed_state["source_image"]
            # state_target = preprocessed_state["target_image"]
            pi_loss, v_loss = self.train_step(
                # state_source,
                # state_target,
                ds_i[0],
                ds_i[1]["policy"],
                ds_i[1]["value"],
            )

            progbar.add(
                1, values=[("pi_loss", float(pi_loss)), ("v_loss", float(v_loss))]
            )

    def evaluate_single_shot(self, steps: int = None, env=None):
        steps = steps or self.general_config.evaluation_steps
        with progressbar.ProgressBar(max_value=steps) as pb:
            with ThreadPoolExecutor(self.general_config.rollout_threads) as executor:

                def eval_fn():
                    reward = self.evaluate_step(self.model, use_mcts=False, env=env)
                    pb.update(pb.value + 1)
                    return reward

                rewards = [executor.submit(eval_fn) for _ in range(steps)]
        results = [r.result() for r in rewards]
        # results = [r.result() for r in rewards]
        result_dict = {k: [] for k in results[0].keys()}
        for res in results:
            for k in res.keys():
                result_dict[k].append(res[k])
        return result_dict

    def evaluate(self, steps: int = None, env=None):
        steps = steps or self.general_config.evaluation_steps
        with progressbar.ProgressBar(max_value=steps) as pb:
            with ThreadPoolExecutor(self.general_config.rollout_threads) as executor:

                def eval_fn():
                    reward = self.evaluate_step(self.model, use_mcts=True, env=env)
                    pb.update(pb.value + 1)
                    return reward

                rewards = [executor.submit(eval_fn) for _ in range(steps)]
        results = [r.result() for r in rewards]
        result_dict = {k: [] for k in results[0].keys()}
        for res in results:
            for k in res.keys():
                result_dict[k].append(res[k])
        return result_dict

    def predict(self, state, single_shot: bool = False, env=None):
        env = env or self.test_env
        if not single_shot:
            # mcts = EvalMCTS(env, self.model, node_selection="best")
            # root, info = mcts.run(model, state)

            mcts = EvalTree(env, self.model)
            root, info = mcts.run(state)

        else:
            mcts = EvalMCTS(env, self.model, node_selection="last")
            root, info = mcts.run(self.model, state)

        return info  # steps_needed, len(state.original_transformations)

    def evaluate_step(self, model, use_mcts=False, env=None):
        # state = self.game.get_init_board()
        # env = self.val_env if split == "val" else self.test_env
        env = env or self.test_env
        # generator = self.val_generator if split == "val" else self.test_generator
        # generator = self.generator_class()
        # env = ITSREnv(generator=generator, preprocessor=self.preprocessor)

        state = env.reset()
        if use_mcts:
            # mcts = EvalMCTS(env, self.model, node_selection="best")
            # root, info = mcts.run(model, state)

            mcts = EvalTree(env, self.model)
            root, info = mcts.run(state)

        else:
            mcts = EvalMCTS(env, self.model, node_selection="last")
            root, info = mcts.run(model, state)

        return info  # steps_needed, len(state.original_transformations)

    def loss_pi(self, targets, outputs):
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            # reduction=tf.keras.losses.Reduction.MEAN,
        )
        return tf.math.reduce_mean(loss_fn(targets, outputs))
        # loss = -tf.math.reduce_sum(
        #     tf.cast(targets, tf.float16) * tf.math.log(tf.maximum(outputs, 1e-5)),
        #     axis=1,
        # )
        # return tf.math.reduce_mean(loss)

    def loss_v(self, targets, outputs):
        # loss = tf.math.reduce_sum((targets-outputs)**2)/targets.size()[0]
        loss = tf.math.reduce_mean((tf.cast(targets, tf.float16) - outputs) ** 2)
        return loss
