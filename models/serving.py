import numpy as np
import torch

# import tensorflow as tf
import threading
import multiprocessing
import time

from models.preprocessors.preprocessor import Preprocessor


class SimpleModel:
    def __init__(self, model, preprocessor: Preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, data):
        batch_data = self.preprocessor.preprocess_batch([data])
        actions, values = self.model.predict_on_batch(batch_data)
        return np.squeeze(actions), np.squeeze(values)


class DummyModel:
    def __init__(self, model, preprocessor: Preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, data):
        return np.random.uniform(0, 1, 5), np.random.uniform(0, 1, 1)


class ModelServerMP:
    def __init__(
        self, model, preprocessor: Preprocessor, batch_size: int, timeout_ms: int = 10
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms

        #  context = multiprocessing.get_context("spawn")
        multiprocessing.set_start_method("spawn")

        self.data_manager = multiprocessing.Manager()
        self.thread_conds = self.data_manager.dict()
        self.data = self.data_manager.dict()
        self.preds = self.data_manager.dict()
        self.running = self.data_manager.Value("b", False)

        self.inference_thread = None

        # self.start()

        # self.last_infernece = None

    def start(self):
        self.running.value = True
        self.inference_thread = torch.multiprocessing.Process(
            target=self.__inference_loop,
        )
        self.inference_thread.start()

    def stop(self):
        self.running.value = False
        self.inference_thread.join()
        self.inference_thread = None

    def predict(self, data):
        thread_id = threading.get_native_id()
        thread_cond = threading.Condition()

        self.data[thread_id] = data

        with thread_cond:
            self.thread_conds[thread_id] = thread_cond
            thread_cond.wait()

        result = self.preds.pop(thread_id)

        return result

    def __inference_loop(self):
        self.last_inference = time.perf_counter()
        while self.running.value:
            time_diff = time.perf_counter() - self.last_inference

            threads = list(self.data.keys())[: self.batch_size]

            if len(threads) == 0:
                continue
            elif (
                len(threads) < self.batch_size and (time_diff * 1000) < self.timeout_ms
            ):
                time.sleep(self.timeout_ms / 1000)
                continue
            batch_data = [self.data.pop(t) for t in threads]

            batch_data = self.preprocessor.preprocess_batch(batch_data)
            with torch.no_grad():
                preds = self.model(**batch_data)

            # del batch_data

            # print(preds.shape)
            if not isinstance(preds, tuple):
                preds = (preds,)

            preds = [p.cpu().detach().numpy() for p in preds]

            for thread, *pred in zip(threads, *preds):
                self.preds[thread] = pred

            while len(threads) > 0:
                thread = threads.pop()
                if thread in self.thread_conds:
                    cond = self.thread_conds.pop(thread)
                    with cond:
                        cond.notify()
                else:
                    print(batch_data, threads, self.preds)
                    threads.append(thread)
            self.last_inference = time.perf_counter()


class ModelServer:
    def __init__(
        self, model, preprocessor: Preprocessor, batch_size: int, timeout_ms: int = 10
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms

        # context = multiprocessing.get_context("spawn")

        # self.data_manager = multiprocessing.Manager()
        self.thread_conds = dict()
        self.data = dict()
        self.preds = dict()
        self.running = False

        self.inference_thread = None

        # self.start()

        # self.last_infernece = None

    def start(self):
        self.running = True
        self.inference_thread = threading.Thread(
            target=self.__inference_loop,
        )
        self.inference_thread.start()

    def stop(self):
        self.running = False
        self.inference_thread.join()
        self.inference_thread = None

    def predict(self, data):
        thread_id = threading.get_native_id()
        thread_cond = threading.Condition()

        self.data[thread_id] = data

        with thread_cond:
            self.thread_conds[thread_id] = thread_cond
            thread_cond.wait()

        result = self.preds.pop(thread_id)

        return result

    def __inference_loop(self):
        self.last_inference = time.perf_counter()
        while self.running:
            time_diff = time.perf_counter() - self.last_inference

            threads = list(self.data.keys())[: self.batch_size]

            if len(threads) == 0:
                continue
            # elif (
            #     len(threads) < self.batch_size and (time_diff * 1000) < self.timeout_ms
            # ):
            #     print(len(threads), "slept")
            #     time.sleep(self.timeout_ms / 1000)
            #     continue
            batch_data = [self.data.pop(t) for t in threads]

            batch_data = self.preprocessor.preprocess_batch(batch_data)
            with torch.no_grad():
                preds = self.model(**batch_data, training=False)

            # print(preds.shape)
            if not isinstance(preds, tuple):
                preds = (preds,)

            preds = [p.cpu().detach().numpy() for p in preds]

            for thread, *pred in zip(threads, *preds):
                self.preds[thread] = pred

            while len(threads) > 0:
                thread = threads.pop()
                if thread in self.thread_conds:
                    cond = self.thread_conds.pop(thread)
                    with cond:
                        cond.notify()
                else:
                    print(batch_data, threads, self.preds)
                    threads.append(thread)
            self.last_inference = time.perf_counter()


# class ModelServer:
#     def __init__(
#         self, model_path, preprocessor: Preprocessor, batch_size: int, timeout_ms: int = 10
#     ):
#         # self.model = model
#         self.model_path = model_path
#         self.preprocessor = preprocessor
#         self.batch_size = batch_size
#         # self.timeout_ms = timeout_ms
#
#         context = multiprocessing.get_context("spawn")
#         self.inference_process = context.Process(target=self.__inference_loop, )
#
#         self.data_manager = multiprocessing.Manager()
#         self.thread_conds = self.data_manager.dict()
#         self.data = self.data_manager.dict()
#         self.preds = self.data_manager.dict()
#         self.running = self.data_manager.Value('b', False)
#
#         self.start()
#
#         # self.last_infernece = None
#
#
#     def start(self):
#         self.running.value = True
#         self.inference_process.start()
#
#     def stop(self):
#         self.running.value = False
#         self.inference_process.join()
#
#     def predict(self, data):
#         thread_id = threading.get_native_id()
#         thread_cond = self.data_manager.Condition()
#
#         self.data[thread_id] = data
#
#         with thread_cond:
#             self.thread_conds[thread_id] = thread_cond
#             thread_cond.wait()
#
#         result = self.preds.pop(thread_id)
#
#         return result
#
#     def __inference_loop(self):
#         model = tf.keras.models.load_model(self.model_path)
#
#         while self.running.value:
#             time_diff = self.perf_counter() * 1000 - self.last_inference
#             threads = list(self.data.keys())[: self.batch_size]
#
#             if len(threads) == 0:
#                 continue
#             elif len(threads) < self.batch_size and time_diff < self.timeout_ms:
#                 continue
#             batch_data = np.stack([self.data.pop(t) for t in threads])
#
#             # batch_data = self.preprocessor.preprocess_batch(batch_data)
#             preds = model.predict_on_batch(batch_data)
#
#             if not isinstance(preds, list):
#                 preds = [preds]
#
#             for thread, *pred in zip(threads, *preds):
#                 self.preds[thread] = pred
#
#             while len(threads) > 0:
#                 thread = threads.pop()
#                 if thread in self.thread_conds:
#                     cond = self.thread_conds.pop(thread)
#                     with cond:
#                         cond.notify()
#                 else:
#                     print(batch_data, threads, self.preds)
#                     threads.append(thread)
#             self.last_inference = time.perf_counter()
#
