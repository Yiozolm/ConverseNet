| metric | fwd 512 pad 1 | fwd 512 pad 0 | bwd 512 pad 1 | bwd 512 pad 0 |
|---|---:|---:|---:|---:|
| time us | 285.5 | 282.0 | 578.3 | 988.2 |
| shared wavefronts (k) | 9922 | 9512 | 15189 | 14563 |
| shared ld bank conflicts (k) | 1041 | 1113 | 1528 | 1630 |
| shared st bank conflicts (k) | 1195 | 723 | 1766 | 1053 |
| instructions (M) | 54.79 | 52.36 | 142.59 | 136.55 |
| stall mio throttle | 0.91 | 1.14 | 0.59 | 0.60 |
| stall barrier | 0.44 | 0.50 | 0.36 | 0.38 |
| stall short scoreboard | 0.87 | 0.74 | 0.53 | 0.65 |
| warp latency / instruction | 7.81 | 8.04 | 6.17 | 6.25 |
| registers / thread | 64 | 79 | 119 | 119 |
| dynamic smem B | 80800 | 80000 | 80800 | 80000 |
