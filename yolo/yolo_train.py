from ultralytics import YOLO
from pathlib import Path
import yaml
import torch


if __name__ == "__main__":
    classes = {
        0: "cube",
        1: "sphere"
    }

    root = Path("spheres_and_cubes")

    config = {
        "path": str(root.absolute()),
        "train": str((root / "images" / "train").absolute()),
        "val": str((root / "images" / "val").absolute()),
        "nc": len(classes),
        "names": classes
    }

    dataset_yaml_path = root / "dataset.yaml"
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"dataset.yaml создан: {dataset_yaml_path}")

    model = YOLO("yolo26n.pt")

    result = model.train(
        data=str(dataset_yaml_path),
        imgsz=512,
        batch=8,
        workers=0,
        epochs=30,
        patience=10,
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=3,
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        degrees=5.0,
        scale=0.5,
        translate=0.1,
        conf=0.001,
        iou=0.7,
        project="runs",
        name="cube_sphere_yolo",
        save=True,
        save_period=5,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=True,
        plots=True,
        val=True,
        close_mosaic=10,
        amp=True
    )

    print("\nDONE")
    print("Результаты сохранены в:", result.save_dir)
    print("Лучшие веса:")
    print(Path(result.save_dir) / "weights" / "best.pt")