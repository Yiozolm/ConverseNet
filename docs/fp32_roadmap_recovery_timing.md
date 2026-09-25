# Recorded failure recovery timing

Original before43 status remains **failed**; its exact checkpoint at update 3000 is recovered without replay.

- Failed-process wall: 1762.548218 s.
- Recovered-process wall: 589.659853 s.
- Process total: 2352.208071 s.
- Supervisor failure-exit event: 2026-09-25T10:32:13.701338+00:00.
- Recovery affinity-wrapper start: 2026-09-25T10:36:55.8985294Z.
- Separate observed restart interval: 282.197191 s (4.703 min).

Original failed status remains failed; no completed_utc is fabricated.
The observed restart interval uses supervisor exit and affinity-wrapper start timestamps, not nonexistent failed-run completion metadata.
Summed process time includes the complete failed process and complete recovered process, including repeated setup. It excludes the restart interval and is not elapsed time including human recovery waits.
The event interval and process timers have different start/end scopes; they are reported separately, without presenting their sum as an exact time-to-quality measurement.
