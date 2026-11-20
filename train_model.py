import argparse
import json
from pathlib import Path
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def main():
    parser = argparse.ArgumentParser(description="Train AgroVision AI MobileNetV2 model.")
    parser.add_argument("--data", default="data/plantvillage", help="Folder containing one subfolder per class.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise SystemExit(
            f"Dataset folder not found: {data_dir}\n"
            "Extract PlantVillage so the folder contains class subfolders, then run again."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_checkpoint.keras", monitor="val_accuracy", save_best_only=True
        ),
    ]
    Path("models").mkdir(exist_ok=True)

    print("\nStage 1: training classification head")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    if args.fine_tune_epochs > 0:
        print("\nStage 2: fine-tuning top layers")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=callbacks)

    model_path = Path("models/agrovision_mobilenetv2.keras")
    model.save(model_path)
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {accuracy * 100:.2f}%")
    print(f"Saved model to: {model_path.resolve()}")
    print("Saved class labels to: models/class_names.json")
    print("Note: use a separate real-world test set before making performance claims.")

if __name__ == "__main__":
    main()
