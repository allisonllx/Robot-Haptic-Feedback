from pathlib import Path

import mujoco

from .config import VIDEO_CAPTURE_EVERY, VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH


class VideoRecorder:
    """Stream offscreen frames to mp4 (uses the same camera as the passive viewer)."""

    def __init__(self, model, path, fps=VIDEO_FPS, width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
        self.path = Path(path)
        self.fps = fps
        self.capture_every = VIDEO_CAPTURE_EVERY
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height, width)
        self._writer = None
        self._frame_counter = 0
        self._saved_frames = 0

    def start(self):
        try:
            import imageio
        except ImportError as exc:
            raise RuntimeError(
                "Video recording requires imageio. Install with: pip install imageio imageio-ffmpeg"
            ) from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            str(self.path),
            fps=self.fps,
            macro_block_size=1,
        )

    def capture(self, data, camera, overlay_callback=None):
        if self._writer is None:
            return

        self._frame_counter += 1
        if (self._frame_counter - 1) % self.capture_every != 0:
            return

        self.renderer.update_scene(data, camera=camera)
        if overlay_callback is not None:
            overlay_callback(self.renderer.scene)
        frame = self.renderer.render()
        self._writer.append_data(frame)
        self._saved_frames += 1

    def close(self):
        if self._writer is None:
            return

        self._writer.close()
        self._writer = None
        if self._saved_frames == 0:
            print("No video frames captured; skipping video save.")
            if self.path.exists():
                self.path.unlink()
            return

        print(
            f"Saved run video ({self._saved_frames} frames @ {self.fps} fps) "
            f"to {self.path.resolve()}"
        )
