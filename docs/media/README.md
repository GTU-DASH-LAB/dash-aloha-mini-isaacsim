# Where the README's clips came from

GIF, not mp4, and not by preference: GitHub does not play a video that lives in a repo.
An `<video src="docs/....mp4">` tag in a README renders as a blank gap with a download
link under it, which is what the top of this README used to be. A GIF renders inline
everywhere — GitHub, GitLab, an editor's Markdown preview, an offline clone.

The sources are the run recordings under `nav/results/videos/`, which are **not in git**
(see [`nav/results/.gitignore`](../../nav/results/.gitignore) — 2.2 GB of them). So these
four files are the only surviving form of these particular runs in the repo, and the
table below is the only record of which run each one is.

| GIF | source run | outcome | encode |
|---|---|---|---|
| `nav_warehouse_aisle05.gif` | `20260831-205547_warehouse.mp4` | **pass**, 1.50 m final over 15.42 m of path, 34.5 s sim, 0 guard stops | both panels, 5×, 960 px, 12 fps, 128 colours |
| `nav_hospital_wheelchairs.gif` | `20260831-204310_hospital_past_wheelchairs.mp4` | **pass**, 1.50 m over 13.94 m, 31.4 s, 0 guard stops | stacked, 5×, 400 px, 8 fps, 80 colours |
| `nav_office_hallway_turn.gif` | `20260831-203539_office_hallway_turn.mp4` | **pass**, 1.50 m over 16.81 m, 55.8 s, 185 guard stops | stacked, 7× |
| `nav_hospital_vending_pivot.gif` | `20260902-024912_hospital_vending_machine__high_pv.mp4` | **pass**, 1.50 m over 12.22 m, 30.9 s, 0 guard stops | stacked, 5× |

The first three are episodes of the **8/13 arc-menu ladder** run on 2026-08-31 evening
(20:16–21:11), so they are three rows of one scored experiment rather than three runs
picked for looking good. The fourth is from `sync_study/pivot3.0_high` and is in the
README only to show what the turn-in-place menu items look like — **do not quote a
comparison off it**: every arm in `sync_study` predates the label-RNG reseed and is
confounded, which [`nav/results/sync_study/README.md`](../../nav/results/sync_study/README.md)
explains.

## Regenerating

The run recording itself comes from the harness:

```bash
python3 nav/tools/make_run_video.py --latest
```

That writes a 2560-wide mp4: menu panel left, third-person right, a caption band beneath
carrying the instruction, the free-space sentence and the model's own reasoning. The
caption band is dropped here — at README width its text is about five pixels tall, so it
would be noise. The instruction is in the README as text instead, where it can be read.

Then, per clip (`SPEED` sets the playback multiple, and it is stated in the README next
to every clip because a sped-up robot is not the robot):

```bash
# hero: both panels side by side, caption band cropped off
ffmpeg -i SRC.mp4 -vf "crop=2560:714:0:0,setpts=PTS/5,fps=12,scale=960:-1:flags=lanczos,\
palettegen=max_colors=128:stats_mode=diff" -y pal.png
ffmpeg -i SRC.mp4 -i pal.png -lavfi "crop=2560:714:0:0,setpts=PTS/5,fps=12,\
scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
  -loop 0 -y out.gif
```

The three small ones stack the panels instead of placing them side by side. That is a
layout decision made by measuring rather than by eye: side by side the frame is 3.58:1,
and in a three-column table GitHub renders each cell at roughly 290 px, which makes the
clip **81 px tall** — the arcs survive as colour and nothing else does. Stacked, the same
290 px of width gives 309 px of height, and the numbered arcs, the chosen path and the
robot are all legible. Check a thumbnail at the size it will actually be displayed, not
at the size you encoded it.

```bash
ffmpeg -i SRC.mp4 -filter_complex \
  "[0:v]crop=1280:714:0:0[m];[0:v]crop=1280:649:1280:65[c];[m][c]vstack=inputs=2,\
setpts=PTS/5,fps=8,scale=400:-1:flags=lanczos,palettegen=max_colors=80:stats_mode=diff" -y pal.png
```

(The chase crop starts at `y=65` to skip the header strip, which is already in the menu
crop above it and would otherwise appear twice.)

Total weight is ~9.5 MB, which is the real constraint on all of this — every clip is
sized to the smallest encode that still reads at its rendered width.
