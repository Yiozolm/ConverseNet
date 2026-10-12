### forward

| metric | forward baseline | forward T=512 | forward T=1024 |
|---|---:|---:|---:|
| time us | 412.6 | 306.9 | 282.0 |
| DRAM read MB | 31.9 | 39.2 | 44.7 |
| DRAM write MB | 17.7 | 18.4 | 18.4 |
| DRAM % peak | - | 41.9 | 50.0 |
| SM % peak | 33.2 | 45.1 | 53.2 |
| global ld sectors (k) | 6261 | 2184 | 2192 |
| global ld requests (k) | 1168 | 333 | 341 |
| L2 read hit % | 44.5 | 58.5 | 54.4 |
| L2 write hit % | 0.3 | 0.0 | 0.4 |
| shared wavefronts (k) | 10260 | 12269 | 12564 |
| instructions (M) | 55.63 | 55.58 | 59.07 |
| local ld instructions | 0 | 0 | 0 |
| warps active % | 33.2 | 33.5 | 66.1 |
| eligible warps / cycle | 0.68 | 0.96 | 2.14 |
| issue active % | 35.0 | 48.0 | 56.6 |
| stall long scoreboard | 4.41 | 1.29 | 1.41 |
| stall short scoreboard | 0.70 | 0.62 | 1.23 |
| stall barrier | 0.80 | 0.65 | 1.86 |
| stall mio throttle | 1.18 | 1.55 | 2.91 |
| stall wait | 1.61 | 1.56 | 1.72 |
| warp latency / instruction | 11.44 | 8.31 | 14.05 |
| registers / thread | 125 | 128 | 64 |
| static smem B | 0 | 1600 | 1600 |

### backward

| metric | backward baseline | backward T=512 | backward T=1024 |
|---|---:|---:|---:|
| time us | 864.8 | 626.2 | 648.6 |
| DRAM read MB | 83.8 | 54.1 | 56.5 |
| DRAM write MB | 57.2 | 59.3 | 74.0 |
| DRAM % peak | - | 40.5 | 45.0 |
| SM % peak | 42.2 | 57.4 | 59.6 |
| global ld sectors (k) | 10329 | 4201 | 4210 |
| global ld requests (k) | 1903 | 647 | 655 |
| L2 read hit % | 39.1 | 78.5 | 83.7 |
| L2 write hit % | 40.6 | 40.9 | 47.1 |
| shared wavefronts (k) | 15511 | 18757 | 19788 |
| instructions (M) | 147.51 | 144.34 | 149.45 |
| local ld instructions | 0 | 0 | 359936 |
| warps active % | 33.4 | 33.1 | 66.7 |
| eligible warps / cycle | 0.92 | 1.56 | 3.12 |
| issue active % | 44.8 | 60.0 | 61.9 |
| stall long scoreboard | 3.52 | 0.36 | 0.64 |
| stall short scoreboard | 0.62 | 0.88 | 1.27 |
| stall barrier | 0.54 | 0.42 | 1.27 |
| stall mio throttle | 0.53 | 0.65 | 1.79 |
| stall wait | 1.13 | 1.02 | 1.47 |
| warp latency / instruction | 8.91 | 6.66 | 12.91 |
| registers / thread | 128 | 128 | 64 |
| static smem B | 128 | 1728 | 1856 |
