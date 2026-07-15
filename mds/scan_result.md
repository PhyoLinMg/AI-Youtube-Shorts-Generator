# Codebase Audit — AI-Youtube-Shorts-Generator

Scope: `shorts_generator/` (pipeline, clipper, webapp, muapi, captions, local/*), `main.py`, `dashboard.py`.
Method: full read of all source files + targeted verification (ffmpeg seek semantics tested empirically, `.env`/git hygiene checked, dependency manifests reviewed).

Threat model for the Security section: this is a **single-user, localhost-only tool** (`dashboard.py` binds `127.0.0.1`, no auth, no debug mode). Findings are rated against that model, not against a multi-tenant/public-internet deployment.

---

## 1. Performance Audit

Ranked by impact. Deep frame-level CPU findings for framing/clipping live in Section 3 — this section covers system/I-O/network-level performance.

### 1.1 `ffmpeg -ss` placement causes full-video decode-and-discard on every clip cut
**File:** `shorts_generator/local/clipper.py:48-57` (`_cut_subclip`)
`-ss` is placed **after** `-i`, which is output seeking: ffmpeg decodes every frame from `t=0` up to `start` and throws it away before it starts encoding the wanted range. For a highlight starting 40 minutes into a 2-hour video, this decodes ~40 minutes of video just to reach the cut point — for every single highlight clipped from that source.
**Fix:** move `-ss` before `-i` (input seeking, keyframe-accurate, no wasted decode). Verified empirically that this changes semantics for `-to`: with `-ss` before `-i`, `-to end` is no longer absolute — it's measured from the seek point, so `-ss 10 -to 15` yields a 15s clip, not 5s. The correct fix is `-ss start -i src -t (end-start)`, not just moving `-ss`.

### 1.2 Three-way re-encode chain per clip
`crop_clip_local` (`local/clipper.py:399-423`) → `burn_captions` (`captions.py:248-289`) chains: (1) ffmpeg libx264 cut, (2) OpenCV `VideoWriter` mp4v reframe encode, (3) ffmpeg libx264 caption burn-in. Same footage gets fully re-encoded three times. See Section 3.3 for the concrete fix (fuse crop into the ffmpeg pass for the default "locked" mode).

### 1.3 Dependency manifests have no upper bounds / lockfile
`requirements.txt` and `requirements-local.txt` use bare `>=` (e.g. `requests>=2.31`). No lockfile (`pip freeze` / `requirements.lock` / `pip-tools`). This is a reproducibility and supply-chain-drift problem, not just style: a `pip install` today and one in six months can resolve to different transitive versions with different performance and security characteristics.
**Fix:** add a lockfile (`pip-compile` or `pip freeze > requirements.lock`) and consider `pip-audit` in CI (see Security 2.5 — don't guess CVEs from memory, run the tool).

### 1.4 `progress.log` grows unbounded across reruns
`run_output.py:250-267` (`capture_progress_log`) opens the log in append mode (`"a"`) and never rotates or truncates. Because `resolve_output_dir` reuses the same folder for the same video title (`run_output.py:121-135`), repeated runs against the same video accumulate one unbounded log file forever. Low severity (text-only), but worth a size cap or truncate-on-run-start if this tool sees long-term repeated use.

### 1.5 Flask dev server, single job slot
`dashboard.py` runs Flask's built-in dev server — fine for a local single-user tool (as documented in `webapp.py`'s module docstring), but flag it explicitly if this ever gets pointed at anything beyond localhost: no worker pool, no production WSGI server, and the one-job-at-a-time design (`_job_lock`) is a hard architectural constraint, not just a current limitation (the module docstring correctly explains why: `capture_progress_log` swaps `sys.stdout`/`sys.stderr` process-globally).

### 1.6 `_title_via_oembed` blocks the request thread on every new video
`run_output.py:49-63` — synchronous `requests.get` (10s timeout) to YouTube's oEmbed endpoint on every `resolve_output_dir` call for a URL not yet seen. Acceptable for a CLI tool; in the webapp path this runs on `start_run`'s request thread before the background thread is spawned (`webapp.py:117-149`), so a slow/hanging oEmbed response delays the HTTP response to the browser by up to 10s. Minor; could be moved into the background thread if it becomes noticeable.

---

## 2. Security Review

(Threat model: local, single-user, `127.0.0.1`-bound Flask app — see header.)

### 2.1 No CSRF protection on state-changing POST routes
**File:** `webapp.py:117-149` (`/run`), `:215-232` (`/history/<name>/delete-source`), `:234-256` (`/history/<name>/delete-shorts`)
These are plain form-accepting POST routes with no CSRF token, no `Origin`/`Referer` check, and no custom-header requirement. A malicious webpage the user has open in the same browser (while the dashboard is running) can auto-submit a hidden form to `http://127.0.0.1:5050/run` or the delete routes — this is a well-known attack class against localhost dev servers (browsers permit cross-origin *form POSTs*, they just block *reading* the response without CORS headers, which this app doesn't send). Concrete impact:
- Trigger an unwanted, resource-heavy pipeline run against an attacker-chosen URL.
- Silently delete the user's own `full_source.mp4` or `Shorts/*.mp4` for any run folder name the attacker can guess/brute-force (folder names are derived from video titles — guessable for known videos).
**Fix:** reject requests whose `Origin`/`Referer` isn't `127.0.0.1`/`localhost`, or require the request body be `application/json` (form auto-submission can't set that content-type without triggering a CORS preflight, which the browser will fail against a server with no CORS headers).

### 2.2 `mode=local` + arbitrary `url` lets `/run` copy other local video files into the output tree
**File:** `local/downloader.py:37-59, 76-114` (`_resolve_local_path`, `download_youtube_local`)
When `mode=local`, the `url` field accepts a bare local filesystem path or `file://` URL, not just a YouTube URL. Combined with 2.1's CSRF gap, an attacker page could POST `mode=local&url=/path/to/some/other/video.mp4` and have the tool copy/remux that file into `output/<title>/full_source.mp4`, later servable via `/download/<name>`. **Scope is bounded, not arbitrary-file-read**: `_ensure_mp4_at` (`local/downloader.py:62-73`) either does a raw `shutil.copyfile` (only when the source filename ends in `.mp4`) or an `ffmpeg -c copy` remux, which requires the source to be a container ffmpeg can demux — a text file (e.g. `id_rsa`) fails ffmpeg's remux and produces nothing. Real impact is limited to disclosure of *other video files* on disk, not arbitrary files. Low severity given the localhost-only binding, but worth closing alongside 2.1 since it's the same trigger vector.

### 2.3 No size cap on downloaded/streamed files
**File:** `shorts_generator/clipper.py:33-41` (`_download_to`)
Streams a hosted clip URL to disk with no max-size check. If `crop_clip` ever returns (or is tricked into returning, via a compromised/misbehaving MuAPI response) an arbitrarily large URL, this fills local disk. Low severity (requires a compromised upstream API or SSRF-adjacent trick), but a cheap `Content-Length` sanity check or streamed size cap is a defense-in-depth win.

### 2.4 Error messages echo raw upstream response bodies
**File:** `muapi.py:34, 54, 81, 85` — `MuAPIError` messages embed `resp.text` / the full response `data` verbatim. If MuAPI ever echoes request data or internal details in error bodies, those land directly in `progress.log` and the dashboard's `/status` log stream, which is served back to the browser. Low severity today (single-user), but avoid habitually piping raw upstream error bodies into user-facing logs.

### 2.5 Don't guess dependency CVEs — run the tool
No pinned versions to audit against a hardcoded CVE list. Recommend wiring `pip-audit` (or `safety`) into CI/pre-release rather than hand-checking version numbers against memorized advisories, which risks both false positives and false negatives.

### What's already solid (worth stating, not just flagging problems)
- **No shell injection surface**: every `subprocess.run`/`subprocess` call across `local/clipper.py`, `captions.py`, `local/downloader.py` uses list-form argv, never `shell=True`. Confirmed by reading every call site.
- **Path traversal is handled correctly**: `webapp.py:100-109` (`_safe_join`) uses `os.path.realpath` + prefix comparison to block `../` escapes, and both `/download` and the `/history/*` routes route through it.
- **Secrets hygiene is correct**: `.env` is gitignored and confirmed untracked (`git ls-files` shows only `.env.example` tracked); API keys are read once via `python-dotenv` into `config.py` and never logged or echoed back to clients.
- **Dashboard binds `127.0.0.1` explicitly** (`dashboard.py:11`), not `0.0.0.0` — the single-user threat model is enforced in code, not just assumed.

---

## 3. CPU / Memory Optimization — Framing & Clipping

This is where the actual per-clip CPU cost lives. Memory is not the bottleneck here — the pipeline streams frames (`cap.read()` → transform → `writer.write()`, no frame buffering beyond `prev_gray` and small O(frame-count) numeric lists for centers/zooms/classes) and has no leak pattern. **CPU — repeated decode passes, per-frame Haar-cascade detection, and a heavyweight resize kernel — is the real cost driver.** The three findings below are the highest-value fixes.

### 3.1 Output-seeking on cut (biggest single cost) — see 1.1
Already covered in Performance 1.1: `-ss` after `-i` means a full decode of everything before the highlight's start time, per clip. This is almost certainly the single largest CPU cost in the whole pipeline for videos where highlights land late in the source. Fix: `-ss` before `-i`, and switch `-to end` → `-t (end-start)`.

### 3.2 Adaptive framing runs face + cursor detection on every single frame, no stride
**File:** `local/clipper.py:178-204` (`_classify_frames`)
Contrast with the default "locked" path, which explicitly strides at `fps // 5` (`local/clipper.py:100`, ~5 detections/sec — "5 frames/sec is plenty" per the docstring at line 98). `_classify_frames` has no equivalent stride: `cv2.CascadeClassifier.detectMultiScale` (Haar cascade — CPU-heavy, no GPU path) plus `_detect_cursor`'s frame-diff/contour pipeline both run on **every** frame of the clip.
**Fix:** classify every Nth frame (same `fps // 5` idea), hold the class/anchor between samples. The existing hysteresis (`_apply_hysteresis`, `MODE_DWELL_SECONDS = 0.75s`) already tolerates coarser sampling than per-frame — dwell time is far larger than the gap between strided samples, so this should cost negligible quality for a ~5x cut in the heaviest per-frame work in the adaptive path.

### 3.3 Locked-mode pass 1 fully decodes every frame to sample 1-in-N
**File:** `local/clipper.py:102-113` (`_reframe_vertical`, pass 1)
`cap.read()` is called every iteration (full decode of every frame) even though the face cascade only runs when `frame_idx % sample_stride == 0` (line 107) — i.e. only 1-in-N decoded frames are actually used.
**Fix:** call `cap.grab()` (decode-free frame advance) on skipped frames and only `cap.retrieve()` (or a fresh `read()`) on sampled frames. Cuts pass-1 decode cost by roughly `sample_stride`x (typically ~5-6x at 30fps).

### 3.4 Double full decode of the same clip (locked mode)
**File:** `local/clipper.py:78-138`
Pass 1 samples for the median face position, then `cap.set(cv2.CAP_PROP_POS_FRAMES, 0)` (line 126) rewinds and pass 2 re-decodes the entire clip to write the crop. The source gets decoded twice on top of the cut in 3.1. This is architecturally required *as currently structured* (the crop box is only known after seeing the whole clip), but see 3.5 for a way to avoid the second decode entirely.

### 3.5 Bigger win (more invasive): replace the OpenCV reframe pass with ffmpeg's `crop` filter for "locked" mode
**File:** `local/clipper.py:125-138`
Once pass 1 determines the static `(x0, y0, crop_w, crop_h)` box, pass 2 currently does per-frame Python/NumPy slicing + `cv2.VideoWriter` mp4v encoding (a second full decode/encode round-trip through OpenCV). Since the crop is a single fixed rectangle for the whole clip, this can be replaced with a single `ffmpeg -i cut.mp4 -vf "crop=w:h:x:y" -c:v libx264 ...` call — no second OpenCV decode, no intermediate `.silent.mp4`, no separate audio-mux step, and ffmpeg's native crop filter is implemented in optimized C rather than a per-frame Python loop. This would collapse pass 2 + the current audio-mux ffmpeg call into one ffmpeg invocation, and cuts one of the three re-encode passes named in Performance 1.2. (Adaptive mode's per-frame moving crop is harder to express as a static ffmpeg filter — keep the OpenCV path there, just apply 3.2/3.6.)

### 3.6 `INTER_LANCZOS4` on every output frame in adaptive mode
**File:** `local/clipper.py:378`
`cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)` runs on every frame of every adaptive-mode clip. Lanczos is the most expensive standard OpenCV interpolation kernel. Since this is downscaling/resizing a crop toward a fixed output size, `cv2.INTER_AREA` (recommended by OpenCV for downscaling) is materially cheaper with comparable visual quality for this use case; `INTER_LINEAR` is cheaper still if quality headroom allows.

### Memory — honest assessment
No meaningful memory-optimization opportunity found. Frames are processed and discarded one at a time (`cap.read()` → transform → `writer.write()`), the retained per-frame state is limited to small numeric lists (`centers`, `zooms`, `modes` — O(frame count) floats/tuples, not frames) and single-frame buffers (`prev_gray`, `frame`). Don't invent a memory-reduction workstream here — the payoff is in CPU (decode/encode pass count and per-frame detection cost), not memory footprint.

---

## Summary — highest-value fixes in priority order
1. **Performance/CPU 1.1 / 3.1** — fix `-ss`/`-i` ordering in `_cut_subclip` (biggest single win, trivial diff, remember the `-to`→`-t` semantics change).
2. **CPU 3.2** — stride the adaptive path's per-frame Haar/cursor detection like the locked path already does.
3. **Security 2.1** — add Origin/Referer (or JSON-content-type) checks to `/run` and the `/history/*/delete-*` routes.
4. **CPU 3.3** — use `cap.grab()` for skipped frames in locked-mode pass 1.
5. **CPU 3.6** — swap `INTER_LANCZOS4` → `INTER_AREA` in the adaptive resize.
6. **CPU/Performance 3.5 / 1.2** — fuse the locked-mode reframe into an ffmpeg `crop` filter pass (bigger refactor, biggest remaining encode-count reduction).
7. **Performance 1.3 / Security 2.5** — add a lockfile and wire up `pip-audit`.
