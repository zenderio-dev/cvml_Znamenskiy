import json
import os

from train_model import train_model


DATA_ROOT = "data/spheres_and_cubes_new/spheres_and_cubes_new"
OUTPUT_DIR = "outputs"

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0003
NUM_WORKERS = 0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models = ["b0", "b1", "b2"]
    results = []

    for model_name in models:
        result = train_model(
            model_name=model_name,
            data_root=DATA_ROOT,
            output_dir=OUTPUT_DIR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            num_workers=NUM_WORKERS
        )
        results.append(result)

    best_model = max(results, key=lambda x: x["accuracy"])

    summary = {
        "results": results,
        "best_model": best_model["model"],
        "best_accuracy": best_model["accuracy"],
        "conclusion": (
            f"Лучше всего справилась EfficientNet-{best_model['model'].upper()}, "
            f"потому что у неё самая высокая точность на валидации: "
            f"{best_model['accuracy']:.4f}"
        )
    }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    for result in results:
        print(
            f"EfficientNet-{result['model'].upper()}: "
            f"{result['accuracy']:.4f} | "
            f"{result['confusion_matrix_path']}"
        )

    print(summary["conclusion"])


if __name__ == "__main__":
    main()