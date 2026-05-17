from model_trainer import train_model


def print_progress(message):
    print(f"[INFO] {message}")


if __name__ == "__main__":
    result = train_model(progress_callback=print_progress)

    print("[INFO] Training complete.")
    print(f"[INFO] Total images: {result['total_images']}")
    print(f"[INFO] Total encodings: {result['total_encodings']}")
    print(f"[INFO] Skipped images: {result['skipped_images']}")
    print(f"[INFO] Saved to: {result['encodings_path']}")