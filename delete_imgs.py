from pathlib import Path


def delete_png_files_in_img_folder() -> None:
    project_root = Path(__file__).resolve().parent
    img_dir = project_root / "img"

    if not img_dir.exists() or not img_dir.is_dir():
        print(f"Folder not found: {img_dir}")
        return

    deleted_count = 0
    for file_path in img_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() == ".png":
            file_path.unlink()
            deleted_count += 1

    print(f"Deleted {deleted_count} PNG file(s) from: {img_dir}")


if __name__ == "__main__":
    delete_png_files_in_img_folder()
