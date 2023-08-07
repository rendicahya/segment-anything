import json
import pathlib

import click
import torch
from moviepy.editor import VideoFileClip
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)

    return image.permute(2, 0, 1).contiguous()


@click.command()
@click.argument(
    "dataset-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "json-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument("threshold", nargs=1, required=False, type=float, default=0.2)
def main(dataset_path, json_path, threshold):
    sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b_01ec64.pth").to(device="cuda")
    n_files = sum(1 for f in dataset_path.glob("**/*") if f.is_file())
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    with open(
        "/nas.dbms/randy/projects/UniDet/datasets/label_spaces/learned_mAP.json", "r"
    ) as f:
        unified_label_file = json.load(f)

    with open("/nas.dbms/randy/projects/UniDet/ucf101_relevant_ids.json", "r") as f:
        relevant_ids = json.load(f)

    thing_classes = [
        "{}".format([xx for xx in x["name"].split("_") if xx != ""][0])
        for x in unified_label_file["categories"]
    ]

    common_obj = "Person", "Man", "Woman"
    common_ids = [thing_classes.index(i) for i in thing_classes if i in common_obj]

    with tqdm(total=n_files) as bar:
        for action in dataset_path.iterdir():
            relevant_objects = [*relevant_ids[action.name], *common_ids]

            for video_file in action.iterdir():
                json_file = json_path / action.name / video_file.with_suffix('.json').name

                if not json_file.exists():
                    print("File not exist:", json_file.name)
                    continue

                bar.set_description(video_file.name)
                video = VideoFileClip(str(video_file))

                with open(json_file, "r") as f:
                    json_data = json.load(f)

                batched_input = []

                for i, frame in enumerate(video.iter_frames()):
                    if str(i) not in json_data.keys():
                        continue

                    image_boxes = []

                    for box, confidence, class_id in json_data[str(i)]:
                        if confidence < threshold or class_id not in relevant_objects:
                            continue

                        # x1, y1, x2, y2 = [round(i) for i in box]
                        image_boxes.append([round(i) for i in box])

                    image_boxes = torch.tensor(image_boxes, device=sam.device)

                    sam_input = {
                        "image": prepare_image(frame, resize_transform, sam),
                        "boxes": resize_transform.apply_boxes_torch(
                            image_boxes, frame.shape[:2]
                        ),
                        "original_size": frame.shape[:2],
                    }

                    batched_input.append(sam_input)

                batched_output = sam(batched_input, multimask_output=False)
                bar.update(1)


if __name__ == "__main__":
    main()
