"""Command-line interface.

The original script had no interface at all: the input file name, the output
name, and every threshold were literals in the source, so trying a second video
meant editing the code.  Everything is a flag here, and the defaults are chosen
to work without any flags at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .config import TABLE_PRESETS, Config
from .pipeline import RunOptions, build_pipeline, run


def _log(msg: str) -> None:
    print(f"[billiards] {msg}", flush=True)


def _parse_corners(text: Optional[str]) -> Optional[List[List[float]]]:
    if not text:
        return None
    parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 8:
        raise argparse.ArgumentTypeError(
            "--table-corners needs 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4"
        )
    nums = [float(p) for p in parts]
    return [nums[i : i + 2] for i in range(0, 8, 2)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="billiards",
        description="Track every ball on a pool table, with no per-video colour tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # -- track -------------------------------------------------------------
    t = sub.add_parser("track", help="track a video (default command)")
    t.add_argument("video", help="input video file")
    t.add_argument("-o", "--output", help="annotated output video (.mp4/.avi)")
    t.add_argument("--csv", dest="export_csv", help="per-frame track positions")
    t.add_argument("--json", dest="export_json", help="run summary and event log")
    t.add_argument("--show", action="store_true", help="display a live window")
    t.add_argument("--debug", action="store_true", help="also show the detection mask")
    t.add_argument("-c", "--config", help="YAML/JSON config file")
    t.add_argument("--preset", choices=sorted(TABLE_PRESETS), help="table preset")
    t.add_argument("--start", type=float, default=0.0, help="start time, seconds")
    t.add_argument("--end", type=float, help="end time, seconds")
    t.add_argument("--max-width", type=int, help="downscale input to this width")
    t.add_argument(
        "--table-corners",
        help="override auto table detection: x1,y1,x2,y2,x3,y3,x4,y4 in pixels",
    )
    t.add_argument("--no-overhead", action="store_true", help="hide the overhead panel")
    t.add_argument(
        "--overhead-inset",
        action="store_true",
        help="put the overhead panel back inside the picture, in a corner, "
             "instead of in a bar below it; keeps the source resolution, at "
             "the cost of covering part of the table",
    )
    t.add_argument("--quiet", action="store_true", help="suppress progress output")

    # -- calibrate ---------------------------------------------------------
    c = sub.add_parser(
        "calibrate",
        help="measure cloth colour and table corners, and report them",
    )
    c.add_argument("video")
    c.add_argument("-c", "--config")
    c.add_argument("--preset", choices=sorted(TABLE_PRESETS))
    c.add_argument("--start", type=float, default=0.0)
    c.add_argument("--end", type=float)
    c.add_argument("--max-width", type=int)
    c.add_argument(
        "--save-preview",
        help="write a PNG showing the detected table, mask and overhead view",
    )
    c.add_argument("--json", dest="export_json", help="write the report as JSON")

    # -- app ---------------------------------------------------------------
    a = sub.add_parser(
        "app",
        help="open the app in a web browser: footage library, set-up, runs, "
             "results and live tracking",
    )
    a.add_argument(
        "--workspace", default="billiards-workspace",
        help="where the app keeps settings, runs and uploads",
    )
    a.add_argument(
        "--folder", action="append", default=[],
        help="list the videos in this folder too (repeatable); the current "
             "folder is listed when none is given",
    )
    a.add_argument("--host", default="127.0.0.1",
                   help="address to listen on; 0.0.0.0 makes it reachable from "
                        "other computers on the network")
    a.add_argument("--port", type=int, default=8765, help="0 picks any free port")
    a.add_argument("--no-browser", action="store_true", help="do not open a browser")
    a.add_argument("--parallel", type=int, default=1,
                   help="how many videos to track at once")

    # -- dump-config -------------------------------------------------------
    d = sub.add_parser("dump-config", help="write the default config to a file")
    d.add_argument("path", help="destination (.yaml or .json)")
    d.add_argument("--preset", choices=sorted(TABLE_PRESETS))

    return p


def _load_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(getattr(args, "config", None))
    if getattr(args, "preset", None):
        cfg.table.preset = args.preset
        # Reset dimensions so the new preset takes effect.
        from .config import TableConfig

        defaults = TableConfig()
        cfg.table.length_in = defaults.length_in
        cfg.table.width_in = defaults.width_in
        cfg.table.ball_diameter_in = defaults.ball_diameter_in
        cfg.apply_preset()
    if getattr(args, "max_width", None):
        cfg.max_frame_width = args.max_width
    if getattr(args, "no_overhead", False):
        cfg.render.overhead_panel = False
    if getattr(args, "overhead_inset", False):
        cfg.render.overhead_panel_place = "inset"
    return cfg


def _time_to_frames(cfg: Config, video: str, start_s: float, end_s: Optional[float]):
    from .video import probe

    info = probe(video)
    start = int(round(max(0.0, start_s) * info.fps))
    end = int(round(end_s * info.fps)) if end_s is not None else None
    return start, end, info


def cmd_track(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    start, end, _ = _time_to_frames(cfg, args.video, args.start, args.end)

    opts = RunOptions(
        video=args.video,
        output=args.output,
        export_csv=args.export_csv,
        export_json=args.export_json,
        show=args.show,
        debug=args.debug,
        start_frame=start,
        end_frame=end,
        table_corners=_parse_corners(args.table_corners),
        on_progress=None if args.quiet else _log,
    )
    summary = run(cfg, opts)

    if not args.quiet:
        _log(
            "done: {} frames in {}s ({} fps)".format(
                summary["frames_processed"],
                summary["wall_seconds"],
                summary["processing_fps"],
            )
        )
        repeated = summary.get("frames_repeated") or 0
        if repeated:
            _log(
                "{} of them repeated the frame before them and were replayed, "
                "not measured twice".format(repeated)
            )
        _log(f"tracks created: {summary['tracks_created']}")
        _log(f"events: {summary['events'] or 'none'}")
        shots = summary.get("shot_log") or []
        if shots:
            _log(f"{len(shots)} shot(s):")
            for shot in shots:
                _log(f"  {shot['summary']}")
        for key in ("output_video", "output_csv", "output_json"):
            if key in summary:
                _log(f"{key}: {summary[key]}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    import cv2

    cfg = _load_config(args)
    start, end, info = _time_to_frames(cfg, args.video, args.start, args.end)
    opts = RunOptions(
        video=args.video, start_frame=start, end_frame=end, on_progress=_log
    )
    pipeline, calib, info = build_pipeline(cfg, opts)

    report = {
        "video": info.to_dict(),
        "calibration": calib.to_dict(),
        "derived": {
            "ball_radius_px_at_table_centre": round(
                calib.table.expected_ball_radius_px(
                    tuple(calib.table.corners_image.mean(axis=0))
                ),
                2,
            ),
            "association_gate_in_per_frame": round(
                cfg.tracker.max_speed_in_s / max(info.fps, 1.0)
                + cfg.tracker.gate_padding_ball_diameters * cfg.table.ball_diameter_in,
                2,
            ),
            "contact_distance_in": round(
                cfg.events.contact_distance_ball_diameters * cfg.table.ball_diameter_in,
                2,
            ),
        },
    }
    print(json.dumps(report, indent=2))
    if args.export_json:
        from .video import write_json

        write_json(args.export_json, report)
        _log(f"wrote {args.export_json}")

    if args.save_preview:
        from .video import sample_frames

        frames = sample_frames(
            args.video, 3, start_frame=start, end_frame=end,
            max_width=cfg.max_frame_width,
        )
        frame = frames[len(frames) // 2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        overlay = frame.copy()
        poly = calib.table.bed_polygon_image(0.0).astype(np.int32)
        cv2.polylines(overlay, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)
        for i, pt in enumerate(calib.table.corners_image.astype(int)):
            cv2.circle(overlay, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(overlay, "TL TR BR BL".split()[i], tuple(pt + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        detections = pipeline.detector.detect(frame, hsv)
        for d in detections:
            cv2.circle(
                overlay,
                (int(d.centre_image[0]), int(d.centre_image[1])),
                int(d.radius_px), (0, 255, 0), 2, cv2.LINE_AA,
            )

        fg = pipeline.detector.foreground_mask(frame, hsv)
        cloth = calib.cloth.mask(hsv)

        from .render import debug_panel

        panel = debug_panel(
            {
                f"table + {len(detections)} detections": overlay,
                "cloth mask (auto-measured)": cloth,
                "foreground = bed AND NOT cloth": fg,
            },
            width=900,
        )
        Path(args.save_preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.save_preview, panel)
        _log(f"wrote {args.save_preview}")
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    from .app import serve

    folders = tuple(Path(f) for f in args.folder) or (Path.cwd(),)
    return serve(
        Path(args.workspace), host=args.host, port=args.port,
        open_browser=not args.no_browser, folders=folders, parallel=args.parallel,
    )


def cmd_dump_config(args: argparse.Namespace) -> int:
    cfg = Config()
    if args.preset:
        cfg.table.preset = args.preset
    cfg.apply_preset()
    cfg.dump(args.path)
    _log(f"wrote {args.path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Allow `billiards clip.mp4 -o out.mp4` as shorthand for the track command.
    known = {"track", "calibrate", "dump-config", "app"}
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        argv.insert(0, "track")

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2

    try:
        if args.command == "track":
            return cmd_track(args)
        if args.command == "calibrate":
            return cmd_calibrate(args)
        if args.command == "dump-config":
            return cmd_dump_config(args)
        if args.command == "app":
            return cmd_app(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"[billiards] error: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
