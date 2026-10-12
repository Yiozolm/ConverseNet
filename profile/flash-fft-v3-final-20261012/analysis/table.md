### forward

| metric | forward baseline | forward v2 T=1024 | forward T=1024 |
|---|---:|---:|---:|
| time us | 412.6 | 282.0 | 277.1 |
| DRAM read MB | 31.9 | 44.7 | 39.6 |
| DRAM write MB | 17.7 | 18.4 | 15.1 |
| DRAM % peak | - | 50.0 | 44.2 |
| SM % peak | 33.2 | 53.2 | 53.3 |
| global ld sectors (k) | 6261 | 2192 | 2214 |
| global ld requests (k) | 1168 | 341 | 341 |
| L2 read hit % | 44.5 | 54.4 | 58.9 |
| L2 write hit % | 0.3 | 0.4 | 5.7 |
| shared wavefronts (k) | 10260 | 12564 | 11988 |
| instructions (M) | 55.63 | 59.07 | 59.04 |
| local ld instructions | 0 | 0 | 0 |
| warps active % | 33.2 | 66.1 | 65.3 |
| eligible warps / cycle | 0.68 | 2.14 | 2.26 |
| issue active % | 35.0 | 56.6 | 57.4 |
| stall long scoreboard | 4.41 | 1.41 | 1.26 |
| stall short scoreboard | 0.70 | 1.23 | 0.89 |
| stall barrier | 0.80 | 1.86 | 1.67 |
| stall mio throttle | 1.18 | 2.91 | 3.15 |
| stall wait | 1.61 | 1.72 | 1.73 |
| warp latency / instruction | 11.44 | 14.05 | 13.86 |
| registers / thread | 125 | 64 | 56 |
| static smem B | 0 | 1600 | 1536 |

### backward

| metric | backward baseline | backward v2 T=512 | backward T=512 |
|---|---:|---:|---:|
| time us | 864.8 | 626.2 | 574.3 |
| DRAM read MB | 83.8 | 54.1 | 59.3 |
| DRAM write MB | 57.2 | 59.3 | 55.9 |
| DRAM % peak | - | 40.5 | 44.8 |
| SM % peak | 42.2 | 57.4 | 59.8 |
| global ld sectors (k) | 10329 | 4201 | 4223 |
| global ld requests (k) | 1903 | 647 | 647 |
| L2 read hit % | 39.1 | 78.5 | 74.9 |
| L2 write hit % | 40.6 | 40.9 | 41.3 |
| shared wavefronts (k) | 15511 | 18757 | 14569 |
| instructions (M) | 147.51 | 144.34 | 136.55 |
| local ld instructions | 0 | 0 | 0 |
| warps active % | 33.4 | 33.1 | 33.3 |
| eligible warps / cycle | 0.92 | 1.56 | 1.73 |
| issue active % | 44.8 | 60.0 | 62.7 |
| stall long scoreboard | 3.52 | 0.36 | 0.40 |
| stall short scoreboard | 0.62 | 0.88 | 0.64 |
| stall barrier | 0.54 | 0.42 | 0.38 |
| stall mio throttle | 0.53 | 0.65 | 0.60 |
| stall wait | 1.13 | 1.02 | 0.96 |
| warp latency / instruction | 8.91 | 6.66 | 6.37 |
| registers / thread | 128 | 128 | 119 |
| static smem B | 128 | 1728 | 1664 |
