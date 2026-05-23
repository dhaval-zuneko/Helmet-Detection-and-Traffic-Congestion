import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


BASE_FRAME_SIZE = (2688, 1520)
BASE_ROIS = {
    "ROI_1": np.array(
        [
            [0, 760],
            [245, 600],
            [1175, 330],
            [1465, 470],
            [980, 1519],
            [0, 1519],
        ],
        dtype=np.float32,
    ),
    "ROI_2": np.array(
        [
            [1320, 470],
            [2485, 205],
            [2687, 235],
            [2687, 1519],
            [1855, 1519],
            [1605, 720],
        ],
        dtype=np.float32,
    ),
}
VEHICLE_CLASS_NAMES = {"bicycle", "car", "motorbike", "motorcycle", "bus", "truck", "auto rickshaw"}
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrackState:
    track_id: int
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    class_name: str
    last_frame_index: int
    first_frame_index: int
    hits: int = 1
    missed: int = 0
    last_speed_px_s: float = 0.0


@dataclass
class RoiWindowStats:
    track_ids: set = field(default_factory=set)
    speed_samples: List[float] = field(default_factory=list)


class SimpleCentroidTracker:
    def __init__(self, max_distance: float, max_missed: int) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_track_id = 1
        self.tracks: Dict[int, TrackState] = {}

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
        fps: float,
    ) -> Dict[int, TrackState]:
        matched_track_ids = set()
        matched_detection_ids = set()

        pairs: List[Tuple[float, int, int]] = []
        track_ids = list(self.tracks.keys())
        for track_id in track_ids:
            track = self.tracks[track_id]
            for det_index, detection in enumerate(detections):
                distance = float(np.linalg.norm(np.array(track.center) - np.array(detection.center)))
                pairs.append((distance, track_id, det_index))

        for distance, track_id, det_index in sorted(pairs, key=lambda item: item[0]):
            if distance > self.max_distance:
                continue
            if track_id in matched_track_ids or det_index in matched_detection_ids:
                continue
            track = self.tracks[track_id]
            detection = detections[det_index]
            frame_gap = max(frame_index - track.last_frame_index, 1)
            speed_px_s = distance * fps / frame_gap
            track.bbox = detection.bbox
            track.center = detection.center
            track.class_name = detection.class_name
            track.last_speed_px_s = speed_px_s
            track.last_frame_index = frame_index
            track.hits += 1
            track.missed = 0
            matched_track_ids.add(track_id)
            matched_detection_ids.add(det_index)

        for det_index, detection in enumerate(detections):
            if det_index in matched_detection_ids:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                center=detection.center,
                class_name=detection.class_name,
                last_frame_index=frame_index,
                first_frame_index=frame_index,
            )

        stale_track_ids: List[int] = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids and track.last_frame_index != frame_index:
                track.missed += 1
            if track.missed > self.max_missed:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            del self.tracks[track_id]

        return dict(self.tracks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuilt Laketown traffic congestion script with hardcoded ROIs."
    )
    parser.add_argument(
        "--input-video",
        default=r"C:\Users\Dhaval Shinde\Downloads\Laketown_TG_Golaghata_Service_Road_second_minute.avi",
        help="Path to the input video.",
    )
    parser.add_argument(
        "--output-dir",
        default="traffic_congestion_laketown_second_minute_rebuilt",
        help="Directory for annotated video and summary files.",
    )
    parser.add_argument(
        "--roi-json",
        default=r"C:\Users\Dhaval Shinde\Desktop\newPOC\traffic_congestion_laketown_1min_yolo11s_conf015\laketown_roi_coordinates.json",
        help="Optional JSON file containing ROI polygons in the original frame coordinate system.",
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Ultralytics model path or model name.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional inference device, for example cpu or 0.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start processing at this second.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Process only this many seconds. Default is full video.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=5.0,
        help="Aggregation window in seconds.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Run detection on every Nth frame.",
    )
    parser.add_argument(
        "--max-track-distance",
        type=float,
        default=120.0,
        help="Maximum centroid distance for track association.",
    )
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=8,
        help="Drop tracks after this many missed frames.",
    )
    parser.add_argument(
        "--high-count-threshold",
        type=int,
        default=15,
        help="Vehicle count threshold for HIGH congestion.",
    )
    parser.add_argument(
        "--medium-count-threshold",
        type=int,
        default=8,
        help="Vehicle count threshold for MEDIUM congestion.",
    )
    parser.add_argument(
        "--high-speed-threshold",
        type=float,
        default=140.0,
        help="Average speed below this becomes HIGH congestion.",
    )
    parser.add_argument(
        "--medium-speed-threshold",
        type=float,
        default=260.0,
        help="Average speed below this becomes MEDIUM congestion.",
    )
    parser.add_argument(
        "--min-track-hits",
        type=int,
        default=2,
        help="Require this many hits before a track is counted.",
    )
    return parser.parse_args()


def load_rois_from_json(roi_json_path: Optional[str]) -> Dict[str, np.ndarray]:
    if not roi_json_path:
        return BASE_ROIS
    path = Path(roi_json_path)
    if not path.exists():
        return BASE_ROIS
    payload = json.loads(path.read_text(encoding="utf-8"))
    rois_payload = payload.get("rois", {})
    loaded: Dict[str, np.ndarray] = {}
    for roi_name, roi_data in rois_payload.items():
        points = roi_data.get("points", [])
        if points:
            loaded[roi_name] = np.array(points, dtype=np.float32)
    return loaded or BASE_ROIS


def scale_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
    base_width, base_height = BASE_FRAME_SIZE
    scale_x = width / base_width
    scale_y = height / base_height
    scaled = points.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled.astype(np.int32)


def center_of_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def detect_vehicles(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    iou: float,
    device: Optional[str],
) -> List[Detection]:
    result = model.predict(
        source=frame,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )[0]
    detections: List[Detection] = []
    if result.boxes is None:
        return detections

    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls.item())
        class_name = str(names.get(cls_id, cls_id)).lower()
        if cls_id not in VEHICLE_CLASS_IDS and class_name not in VEHICLE_CLASS_NAMES:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        bbox = (x1, y1, x2, y2)
        detections.append(
            Detection(
                bbox=bbox,
                center=center_of_bbox(bbox),
                confidence=float(box.conf.item()),
                class_id=cls_id,
                class_name=class_name,
            )
        )
    return detections


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def severity_from_count(count: int, high_count: int, medium_count: int) -> int:
    if count >= high_count:
        return 2
    if count >= medium_count:
        return 1
    return 0


def severity_from_speed(speed_px_s: float, high_speed: float, medium_speed: float) -> int:
    if speed_px_s <= 0:
        return 0
    if speed_px_s <= high_speed:
        return 2
    if speed_px_s <= medium_speed:
        return 1
    return 0


def congestion_label(
    vehicle_count: int,
    avg_speed_px_s: float,
    high_count: int,
    medium_count: int,
    high_speed: float,
    medium_speed: float,
) -> str:
    severity = max(
        severity_from_count(vehicle_count, high_count, medium_count),
        severity_from_speed(avg_speed_px_s, high_speed, medium_speed),
    )
    return ["LOW", "MEDIUM", "HIGH"][severity]


def draw_overlay(
    frame: np.ndarray,
    rois: Dict[str, np.ndarray],
    tracks: Dict[int, TrackState],
    min_track_hits: int,
) -> np.ndarray:
    annotated = frame.copy()
    roi_colors = {"ROI_1": (0, 220, 255), "ROI_2": (255, 180, 0)}
    for roi_name, polygon in rois.items():
        color = roi_colors.get(roi_name, (0, 255, 0))
        cv2.polylines(annotated, [polygon], True, color, 3)
        label_anchor = tuple(polygon[0].tolist())
        cv2.putText(
            annotated,
            roi_name,
            (label_anchor[0] + 10, label_anchor[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

    for track_id, track in tracks.items():
        if track.hits < min_track_hits:
            continue
        center = (int(track.center[0]), int(track.center[1]))
        roi_name = None
        for candidate_name, polygon in rois.items():
            if point_in_polygon(track.center, polygon):
                roi_name = candidate_name
                break
        if roi_name is None:
            continue
        color = roi_colors.get(roi_name, (0, 255, 0))
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, center, 4, color, -1)
        label = f"{track.class_name} #{track_id} {track.last_speed_px_s:.1f}px/s"
        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def build_summary_rows(
    window_stats: Dict[int, Dict[str, RoiWindowStats]],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for window_index in sorted(window_stats.keys()):
        start_second = round(window_index * args.window_sec, 2)
        end_second = round(start_second + args.window_sec, 2)
        for roi_name in ("ROI_1", "ROI_2"):
            stats = window_stats[window_index][roi_name]
            vehicle_count = len(stats.track_ids)
            avg_speed = round(float(np.mean(stats.speed_samples)), 2) if stats.speed_samples else 0.0
            rows.append(
                {
                    "window_index": window_index,
                    "start_second": int(start_second) if start_second.is_integer() else start_second,
                    "end_second": int(end_second) if end_second.is_integer() else end_second,
                    "roi": roi_name,
                    "vehicle_count": vehicle_count,
                    "avg_vehicle_speed_px_s": avg_speed,
                    "congestion_label": congestion_label(
                        vehicle_count=vehicle_count,
                        avg_speed_px_s=avg_speed,
                        high_count=args.high_count_threshold,
                        medium_count=args.medium_count_threshold,
                        high_speed=args.high_speed_threshold,
                        medium_speed=args.medium_speed_threshold,
                    ),
                }
            )
    return rows


def ensure_window(
    window_stats: Dict[int, Dict[str, RoiWindowStats]],
    window_index: int,
) -> Dict[str, RoiWindowStats]:
    if window_index not in window_stats:
        window_stats[window_index] = {"ROI_1": RoiWindowStats(), "ROI_2": RoiWindowStats()}
    return window_stats[window_index]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_stem = input_path.stem
    annotated_path = output_dir / f"{summary_stem}_congestion_annotated.mp4"
    csv_path = output_dir / f"{summary_stem}_congestion_summary.csv"
    json_path = output_dir / f"{summary_stem}_congestion_summary.json"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(int(args.start_sec * fps), 0)
    if args.duration_sec is None:
        end_frame = total_frames
    else:
        end_frame = min(total_frames, start_frame + int(args.duration_sec * fps))

    roi_source = load_rois_from_json(args.roi_json)
    rois = {name: scale_polygon(points, width, height) for name, points in roi_source.items()}

    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {annotated_path}")

    model = YOLO(args.model)
    tracker = SimpleCentroidTracker(
        max_distance=args.max_track_distance,
        max_missed=args.max_missed_frames,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    active_tracks: Dict[int, TrackState] = {}
    window_stats: Dict[int, Dict[str, RoiWindowStats]] = {}

    while frame_index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_index - start_frame) % args.frame_step == 0:
            detections = detect_vehicles(
                model=model,
                frame=frame,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
            )
            active_tracks = tracker.update(detections=detections, frame_index=frame_index, fps=fps)

            current_second = (frame_index - start_frame) / fps
            window_index = int(current_second // args.window_sec)
            current_window = ensure_window(window_stats, window_index)

            for track_id, track in active_tracks.items():
                if track.hits < args.min_track_hits:
                    continue
                for roi_name, polygon in rois.items():
                    if not point_in_polygon(track.center, polygon):
                        continue
                    current_window[roi_name].track_ids.add(track_id)
                    if track.last_speed_px_s > 0:
                        current_window[roi_name].speed_samples.append(track.last_speed_px_s)

        annotated = draw_overlay(
            frame=frame,
            rois=rois,
            tracks=active_tracks,
            min_track_hits=args.min_track_hits,
        )
        writer.write(annotated)
        frame_index += 1

    cap.release()
    writer.release()

    rows = build_summary_rows(window_stats, args)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "window_index",
            "start_second",
            "end_second",
            "roi",
            "vehicle_count",
            "avg_vehicle_speed_px_s",
            "congestion_label",
        ]
        writer_obj = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_obj.writeheader()
        writer_obj.writerows(rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print(f"Annotated video: {annotated_path}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
