import json
import pathlib

import click
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from moviepy.editor import ImageSequenceClip, VideoFileClip
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)

    return image.permute(2, 0, 1).contiguous()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


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
@click.argument(
    "output-path",
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
@click.option("--threshold", nargs=1, required=False, type=float, default=0.2)
def main(dataset_path, json_path, output_path, threshold):
    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth").to(
        device="cuda"
    )
    predictor = SamPredictor(sam)
    n_files = sum(1 for f in dataset_path.glob("**/*") if f.is_file())

    with open("../UniDet/datasets/label_spaces/learned_mAP.json", "r") as f:
        unified_label_file = json.load(f)

    with open("../UniDet/ucf101_relevant_ids.json", "r") as f:
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
                json_file = (
                    json_path / action.name / video_file.with_suffix(".json").name
                )

                if not json_file.exists():
                    print("File not exist:", json_file.name)
                    continue

                video = VideoFileClip(str(video_file))
                n_frames = video.reader.nframes

                output_frames = []
                output_file_path = output_path / action.name / video_file.name
                output_file_path.parent.mkdir(parents=True, exist_ok=True)

                output_mask_path = (
                    pathlib.Path(str(output_path.absolute()) + "-mask")
                    / action.name
                    / video_file.stem
                )
                output_mask_path.mkdir(parents=True, exist_ok=True)

                with open(json_file, "r") as f:
                    json_data = json.load(f)

                for i, image in enumerate(video.iter_frames()):
                    if str(i) not in json_data.keys():
                        continue

                    bar.set_description(f"{video_file.name} ({i}/{n_frames})")

                    predictor.set_image(image)
                    input_boxes = []

                    for box, confidence, class_id in json_data[str(i)]:
                        if confidence < threshold or class_id not in relevant_objects:
                            continue

                        input_boxes.append([round(i) for i in box])

                    input_boxes = torch.tensor(input_boxes, device=sam.device)
                    transformed_boxes = predictor.transform.apply_boxes_torch(
                        input_boxes, image.shape[:2]
                    )

                    if len(input_boxes) == 0:
                        height, width, _ = image.shape
                        black_image = np.zeros((height, width), dtype=np.uint8)

                        imageio.imwrite(output_mask_path / f"{i:04}.png", black_image)
                        output_frames.append(image)

                        continue

                    masks, _, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )

                    masks = masks.cpu().numpy()
                    merged_mask = (
                        np.logical_or.reduce(masks).squeeze(axis=0).astype(np.uint8)
                        * 255
                    )
                    imageio.imwrite(output_mask_path / f"{i:04}.png", merged_mask)

                    plt.figure()
                    plt.imshow(image)

                    for mask in masks:
                        show_mask(mask, plt.gca())

                    for box in input_boxes:
                        show_box(box.cpu().numpy(), plt.gca())

                    plt.axis("off")
                    plt.tight_layout()
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                    canvas = FigureCanvasAgg(plt.gcf())
                    canvas.draw()
                    image_array = np.array(canvas.buffer_rgba())

                    output_frames.append(image_array)

                ImageSequenceClip(output_frames, fps=video.fps).write_videofile(
                    str(output_file_path), audio=False, logger=None
                )

                bar.update(1)


if __name__ == "__main__":
    main()
