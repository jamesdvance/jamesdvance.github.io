---
layout: page
title: Hit & Run Protection On A Home Security Cam
description: Detecting hit & run collisions, LPR, and generating synthetic data for evaluation
img: assets/video/hit_and_run_2.gif
importance: 2
category: side projects
related_publications: false
---

I live two doors down from a bar + barbeque restaurant. I'll never complain about being steps away from pulled pork or smoked ribs, but there's been one unexpected downside: *A nonzero amount of people leavning the bar miss the left turn near my house and then hit my car backing up*

Here's example 1, courtesy of my neighbor's Ring camera: 

{% include video.html path="assets/video/hit_and_run_1.mp4" controls=true width="100%" caption="Example 1: Hit and run caught on security camera" %}

Here's example 2, almost exactly a month later on the same weeknight (happens to be live music night) at the same time of night. (At least this person thought about stopping for a second) 

{% include video.html path="assets/video/hit_and_run_2.mp4" controls=true width="100%" caption="Example 2: Hit and run caught on security camera" %}

I considered investing in an actual license plate recognition camera (LPR) but its $1k + for a single camera and result in some reasonable pushback from privacy-concerned neighbors. So instead, I've rigged up an unused Jetson Nano and bought an empty camera case and created my own smart video monitor. The goal is to have a camera or multiple cameras that can see well enough to reasonably discern a hit-and-run scenario and send me some kind of alert (like pagerduty or a text message). 

  <div class="row">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.html path="assets/img/jetson_in_camera_case.jpg" class="img-fluid rounded
  z-depth-1" zoomable=true %}
      </div>
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.html path="assets/img/camera_on_wall.jpg" class="img-fluid rounded z-depth-1"
  zoomable=true %}
      </div>
  </div>
  <div class="caption">
      Left: Jetson Nano inside the camera enclosure. Right: the camera setup and ready to catch the next knucklehead.
  </div>

## Purpose Build Models Vs VLMs

For video footage, there are two potential ways to build this alerting system. A 'classic', non-language-based model would train on a model that would combine some CV techniques like object detection + trajectory estimation (to greatly simplify. Check out this recent [Kaggle competition](https://www.kaggle.com/competitions/accident/code?competitionId=127283&sortBy=voteCount&excludeNonAccessedDatasources=true) for good examples. ). The second option is to use a Vision Language Model or VLM. The advantage with the VLM is that it doesn't require training at all, can be accessed with a 3rd-party hosted api, and can be adapted to infinite use cases ("check if someone took my parking spot") and also take actions within an agentic workflow. The latter point is really attractive for future use cases (maybe an agent could just play a noise if there's a suspicion of hit and run)

Given VLM's are simpler to startup with and offer much higher flexibility, they're the best choice for this project. 

## Evaluation

Model evaluation is crucial no matter what type of model you're using. But to get an absolute simplest baseline we only need a few examples of a hit and run and a few counterfactuals to determine if it can generally know a hit and run when it sees one. 

### Generating Synthetic Hit-and-Runs

I didn't want to scrape the web endlessly, and anyway I wanted to evaluate on my same angle, perspective and car. With video inpainting, we can segment my real video footage then mask the parts involved in my dataset, while mantaining the real background from my camera setup. 

#### Pipeline

The generation pipeline has three stages, each handled by a separate model:

**1. Object localization (color heuristic).** The first frame is converted to HSV and thresholded for the target color (in this case, the green of the parked car across the street). Morphological open/close cleans up the mask, and the largest contour gives a centroid that becomes the seed point for the next stage. This is a few lines of OpenCV and exists only to avoid manually clicking on the car.

**2. Per-frame mask propagation (SAM2).** The seed point is fed to Meta's [SAM2](https://github.com/facebookresearch/sam2) video predictor (`facebook/sam2-hiera-large`), which propagates a tight binary mask of the car across every frame of the clip. SAM2 handles the parts color thresholding can't — wheels, shadows, partial occlusion — and produces a clean per-frame mask stack saved as both individual PNGs and a packed `.npz`.

**3. Inpainting (VideoPainter + CogVideoX-5B).** The masked frames and a text prompt are passed to [VideoPainter](https://github.com/TencentARC/VideoPainter), which wraps the CogVideoX-5B image-to-video diffusion model with a context encoder branch trained specifically for inpainting. The prompt *"A car crashes into the parked green car, crumpling the front fender and bumper, scattering debris across the pavement, then drives away"* — drives what gets generated inside the mask. Everything outside the mask is preserved bit-for-bit from the source video. Inference runs on a 24GB+ GPU at bfloat16, processing 49 frames per pass at 8 FPS output.

#### Why Inpainting Over Pure Generation

A few properties make inpainting attractive over text-to-video from scratch:

- **Distribution match.** The unaltered background means the evaluation dataset actually matches the 
- **Controllable labels.** The prompt is the label. We can rest assured the label will match the video
- **Scalable variation.** Swapping the prompt produces hit-and-runs of different severities (sideswipe vs. head-on, with vs. without debris), and re-running on different source clips covers different times of day and viewing angles.

#### Evaluation Pair Output

Each run exports a paired sample to `data/hitandrun_eval/`:

- `original.mp4` — the unmodified source clip, serving as the matched negative example
- `hitandrun_synthetic.mp4` — the inpainted clip with the collision
- `metadata.json` — the prompt, seed point, model versions, and human-readable label used for downstream grading

The matched-pair structure matters: a detector that fires on every passing car will flag the negative just as often as the positive, so the per-clip F1 collapses. Pairing forces the alerting system to discriminate the *event*, not just the presence of motion or vehicles.


