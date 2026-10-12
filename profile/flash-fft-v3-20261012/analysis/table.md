### forward

| metric | forward baseline | forward v2 T=1024 | forward T=1024 |
|---|---:|---:|---:|
| time us | 412.6 | 282.0 | 277.3 |
| DRAM read MB | 31.9 | 44.7 | 42.1 |
| DRAM write MB | 17.7 | 18.4 | 18.0 |
| DRAM % peak | - | 50.0 | 48.4 |
| SM % peak | 33.2 | 53.2 | 53.9 |
| global ld sectors (k) | 6261 | 2192 | 2214 |
| global ld requests (k) | 1168 | 341 | 341 |
| L2 read hit % | 44.5 | 54.4 | 57.0 |
| L2 write hit % | 0.3 | 0.4 | 29.9 |
| shared wavefronts (k) | 10260 | 12564 | 11988 |
| instructions (M) | 55.63 | 59.07 | 59.04 |
| local ld instructions | 0 | 0 | 0 |
| warps active % | 33.2 | 66.1 | 63.6 |
| eligible warps / cycle | 0.68 | 2.14 | 2.24 |
| issue active % | 35.0 | 56.6 | 56.6 |
| stall long scoreboard | 4.41 | 1.41 | 1.13 |
| stall short scoreboard | 0.70 | 1.23 | 0.89 |
| stall barrier | 0.80 | 1.86 | 1.67 |
| stall mio throttle | 1.18 | 2.91 | 3.15 |
| stall wait | 1.61 | 1.72 | 1.73 |
| warp latency / instruction | 11.44 | 14.05 | 14.05 |
| registers / thread | 125 | 64 | 56 |
| static smem B | 0 | 1600 | 1536 |

### backward

| metric | backward baseline | backward v2 T=512 | backward T=512 | backward T=768 | backward T=1024 |
|---|---:|---:|---:|---:|---:|
| time us | 864.8 | 626.2 | 613.5 | 620.6 | 617.8 |
| DRAM read MB | 83.8 | 54.1 | 64.1 | 62.7 | 60.6 |
| DRAM write MB | 57.2 | 59.3 | 59.0 | 68.9 | 61.3 |
| DRAM % peak | - | 40.5 | 44.9 | 47.4 | 44.1 |
| SM % peak | 42.2 | 57.4 | 57.5 | 59.4 | 62.0 |
| global ld sectors (k) | 10329 | 4201 | 4223 | 4227 | 4231 |
| global ld requests (k) | 1903 | 647 | 647 | 651 | 655 |
| L2 read hit % | 39.1 | 78.5 | 72.8 | 77.8 | 77.5 |
| L2 write hit % | 40.6 | 40.9 | 42.5 | 43.5 | 41.5 |
| shared wavefronts (k) | 15511 | 18757 | 17798 | 18469 | 18452 |
| instructions (M) | 147.51 | 144.34 | 141.50 | 148.29 | 148.05 |
| local ld instructions | 0 | 0 | 0 | 190976 | 32256 |
| warps active % | 33.4 | 33.1 | 33.4 | 49.5 | 66.0 |
| eligible warps / cycle | 0.92 | 1.56 | 1.60 | 2.30 | 3.51 |
| issue active % | 44.8 | 60.0 | 60.7 | 62.0 | 65.7 |
| stall long scoreboard | 3.52 | 0.36 | 0.40 | 0.64 | 0.58 |
| stall short scoreboard | 0.62 | 0.88 | 0.78 | 1.01 | 0.98 |
| stall barrier | 0.54 | 0.42 | 0.46 | 1.19 | 1.31 |
| stall mio throttle | 0.53 | 0.65 | 0.70 | 1.10 | 1.70 |
| stall wait | 1.13 | 1.02 | 1.01 | 1.01 | 1.12 |
| warp latency / instruction | 8.91 | 6.66 | 6.58 | 9.67 | 12.16 |
| registers / thread | 128 | 128 | 125 | 80 | 64 |
| static smem B | 128 | 1728 | 1664 | 1728 | 1792 |
