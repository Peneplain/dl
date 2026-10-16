# Recorded demonstrations

The 22 selected videos show the original text instructions, execution stages, simulation time and method labels. They include successful and failed trials from both motion generators and paired examples of the learned controller.

[Report and source](https://github.com/Peneplain/dl/tree/main/report) · [Numerical results](https://github.com/Peneplain/dl/blob/main/docs/results/public-results.json) · [Video metadata and SHA-256 hashes](https://github.com/Peneplain/dl/blob/main/videos/manifest.json)

## ARDY and Kimodo generator comparison

Both generator recordings in a row use the same scene.

| Scene | ARDY | Kimodo | View |
|---|---|---|---|
| 60000 | [ARDY · failure](https://github.com/Peneplain/dl/blob/main/videos/baseline100/ardy-seed60000/ardy-seed60000-diagonal45-annotated.mp4) | [Kimodo · success](https://github.com/Peneplain/dl/blob/main/videos/baseline100/kimodo-seed60000/kimodo-seed60000-diagonal45-annotated.mp4) | Elevated 45° |
| 60004 | [ARDY · success](https://github.com/Peneplain/dl/blob/main/videos/baseline100/ardy-seed60004/ardy-seed60004-diagonal45-annotated.mp4) | [Kimodo · failure](https://github.com/Peneplain/dl/blob/main/videos/baseline100/kimodo-seed60004/kimodo-seed60004-diagonal45-annotated.mp4) | Elevated 45° |

## Risk-guided residual controller

Each row pairs the ARDY baseline with Ours. A rescue means baseline failure with controller success; a harm means baseline success with controller failure.

| Category | Scene | ARDY baseline | Ours | View |
|---|---:|---|---|---|
| Shared success | 76912 | [ARDY · success](https://github.com/Peneplain/dl/blob/main/videos/clips/first-both-success-seed76912-ardy-diagonal45.mp4) | [Ours · success](https://github.com/Peneplain/dl/blob/main/videos/clips/first-both-success-seed76912-ours-diagonal45.mp4) | Elevated 45° |
| Rescue | 76916 | [ARDY · failure](https://github.com/Peneplain/dl/blob/main/videos/clips/first-rescue-seed76916-ardy-lateral.mp4) | [Ours · success](https://github.com/Peneplain/dl/blob/main/videos/clips/first-rescue-seed76916-ours-lateral.mp4) | Lateral |
| Rescue | 76948 | [ARDY · failure](https://github.com/Peneplain/dl/blob/main/videos/clips/second-rescue-seed76948-ardy-diagonal45.mp4) | [Ours · success](https://github.com/Peneplain/dl/blob/main/videos/clips/second-rescue-seed76948-ours-diagonal45.mp4) | Elevated 45° |
| Rescue | 76956 | [ARDY · failure](https://github.com/Peneplain/dl/blob/main/videos/clips/third-rescue-seed76956-ardy-diagonal45.mp4) | [Ours · success](https://github.com/Peneplain/dl/blob/main/videos/clips/third-rescue-seed76956-ours-diagonal45.mp4) | Elevated 45° |
| Harm | 77000 | [ARDY · success](https://github.com/Peneplain/dl/blob/main/videos/clips/first-harm-seed77000-ardy-lateral.mp4) | [Ours · failure](https://github.com/Peneplain/dl/blob/main/videos/clips/first-harm-seed77000-ours-lateral.mp4) | Lateral |
| Harm | 77228 | [ARDY · success](https://github.com/Peneplain/dl/blob/main/videos/clips/second-harm-seed77228-ardy-diagonal45.mp4) | [Ours · failure](https://github.com/Peneplain/dl/blob/main/videos/clips/second-harm-seed77228-ours-diagonal45.mp4) | Elevated 45° |
| Harm | 77328 | [ARDY · success](https://github.com/Peneplain/dl/blob/main/videos/clips/third-harm-seed77328-ardy-diagonal45.mp4) | [Ours · failure](https://github.com/Peneplain/dl/blob/main/videos/clips/third-harm-seed77328-ours-diagonal45.mp4) | Elevated 45° |
| Shared failure | 76920 | [ARDY · failure](https://github.com/Peneplain/dl/blob/main/videos/clips/first-both-failure-seed76920-ardy-lateral.mp4) | [Ours · failure](https://github.com/Peneplain/dl/blob/main/videos/clips/first-both-failure-seed76920-ours-lateral.mp4) | Lateral |
| Shared failure | 76924 | [ARDY · failure](https://github.com/Peneplain/dl/blob/main/videos/clips/second-both-failure-seed76924-ardy-diagonal45.mp4) | [Ours · failure](https://github.com/Peneplain/dl/blob/main/videos/clips/second-both-failure-seed76924-ours-diagonal45.mp4) | Elevated 45° |

## Metadata

The public manifest records relative filenames, method, scene seed, example category, complete-trial outcome, camera view, frame rate, frame count, duration, byte size and SHA-256 hash. MP4 bytes match the selected presentation recordings.

These examples illustrate recorded behavior. Quantitative success rates use the complete corresponding experimental cohorts in the numerical results. Packaging creates no new physics steps, policy calls, training or evaluation trials.
