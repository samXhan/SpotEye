from os.path import join as opj
from os.path import splitext


def build_prw_record_paths(
    dataset_dir,
    image_name,
    *,
    image_dir="frames",
    depth_dir="Depth",
    paired_image_dir=None,
    setting_name="PRW-TBPS",
    clean_retain=None,
):
    stem, _ = splitext(image_name)
    record = {
        "file_name": opj(dataset_dir, image_dir, image_name),
        "depth_file_name": opj(dataset_dir, depth_dir, stem + ".png"),
        "image_id": image_name,
        "pairing_key": stem,
        "setting_name": setting_name,
        "clean_retain": clean_retain,
    }
    if paired_image_dir is not None:
        record["paired_clean_file_name"] = opj(dataset_dir, paired_image_dir, image_name)
    return record
