import fire
import utils


def evaluate(conf_name: str):
    model = utils.load_model(conf_name=conf_name, step=None)

    model.load_weights(name='checkpoint_0')

    k_retrievals = model.evaluate_top_k()
    single_shot_accuracy = model.evaluate_single_shot()
    k_values = list(range(10, 101, 5))

    print(f"Single-shot Accuracy: {single_shot_accuracy:.4f}")
    print("Top-K Accuracy:")
    for k in k_values:
        k_accuracy = (k_retrievals <= k).mean()
        print(f" - K={k}: {k_accuracy:.4f}")


if __name__ == "__main__":
    fire.Fire(evaluate)
