import numpy as np

# import tensorflow as tf
import torch
from typing import List, Dict
from torchvision import transforms

from generators.image import ImagePair


class Preprocessor:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        )

    def preprocess(self, pair: ImagePair) -> Dict[str, np.ndarray]:
        source, target = pair.numpy()
        source = self.transform(source)
        target = self.transform(target)
        return {"source_image": source, "target_image": target}

    def preprocess_batch(self, pair_list: List[ImagePair]) -> Dict[str, np.ndarray]:
        output = {"source_image": [], "target_image": []}

        for pair in pair_list:
            preprocessed_pair = self.preprocess(pair=pair)
            output["source_image"].append(preprocessed_pair["source_image"])
            output["target_image"].append(preprocessed_pair["target_image"])

        # return next(tf.data.Dataset.from_tensor_slices(output).batch(len(pair)))

        # output["source_image"] = torch.stack(output["source_image"], axis=0)
        # output["target_image"] = torch.stack(output["target_image"], axis=0)
        output["source_image"] = torch.stack(output["source_image"], axis=0).cuda()
        output["target_image"] = torch.stack(output["target_image"], axis=0).cuda()

        return output
