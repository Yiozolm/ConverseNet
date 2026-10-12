### forward

| metric | forward baseline | forward v3 T=1024 | forward T=768 | forward T=1024 |
|---|---:|---:|---:|---:|
| time us | 412.6 | 277.1 | 766.3 | 557.8 |
| DRAM read MB | 31.9 | 39.6 | 83.7 | 72.5 |
| DRAM write MB | 17.7 | 15.1 | 143.1 | 103.4 |
| DRAM % peak | - | 44.2 | 66.1 | 70.4 |
| SM % peak | 33.2 | 53.3 | 25.5 | 35.8 |
| global ld sectors (k) | 6261 | 2214 | 2166 | 2170 |
| global ld requests (k) | 1168 | 341 | 333 | 337 |
| L2 read hit % | 44.5 | 58.9 | 87.2 | 85.4 |
| L2 write hit % | 0.3 | 5.7 | 74.5 | 80.2 |
| shared wavefronts (k) | 10260 | 11988 | 10520 | 12738 |
| instructions (M) | 55.63 | 59.04 | 57.05 | 59.51 |
| local ld instructions | 0 | 0 | 3039900 | 2139136 |
| warps active % | 33.2 | 65.3 | 56.2 | 66.6 |
| eligible warps / cycle | 0.68 | 2.26 | 0.37 | 1.09 |
| issue active % | 35.0 | 57.4 | 18.2 | 29.8 |
| stall long scoreboard | 4.41 | 1.26 | 13.57 | 5.63 |
| stall short scoreboard | 0.70 | 0.89 | 0.72 | 1.22 |
| stall barrier | 0.80 | 1.67 | 12.81 | 7.46 |
| stall mio throttle | 1.18 | 3.15 | 1.66 | 3.62 |
| stall wait | 1.61 | 1.73 | 1.87 | 1.92 |
| warp latency / instruction | 11.44 | 13.86 | 32.93 | 26.83 |
| registers / thread | 125 | 56 | 40 | 64 |
| static smem B | 0 | 1536 | 1536 | 1536 |

### backward

| metric | backward baseline | backward v3 T=512 | backward T=512 | backward T=768 |
|---|---:|---:|---:|---:|
| time us | 864.8 | 574.3 | 2733.4 | 2752.6 |
| DRAM read MB | 83.8 | 59.3 | 362.4 | 393.4 |
| DRAM write MB | 57.2 | 55.9 | 415.6 | 378.2 |
| DRAM % peak | - | 44.8 | 72.1 | 70.9 |
| SM % peak | 42.2 | 59.8 | 15.1 | 15.1 |
| global ld sectors (k) | 10329 | 4223 | 4179 | 4183 |
| global ld requests (k) | 1903 | 647 | 643 | 647 |
| L2 read hit % | 39.1 | 74.9 | 43.8 | 46.8 |
| L2 write hit % | 40.6 | 41.3 | 68.3 | 46.3 |
| shared wavefronts (k) | 15511 | 14569 | 15188 | 15440 |
| instructions (M) | 147.51 | 136.55 | 147.41 | 149.41 |
| local ld instructions | 0 | 0 | 2737152 | 3273728 |
| warps active % | 33.4 | 33.3 | 31.5 | 53.6 |
| eligible warps / cycle | 0.92 | 1.73 | 0.29 | 0.40 |
| issue active % | 44.8 | 62.7 | 15.4 | 16.2 |
| stall long scoreboard | 3.52 | 0.40 | 10.50 | 12.78 |
| stall short scoreboard | 0.62 | 0.64 | 0.70 | 0.70 |
| stall barrier | 0.54 | 0.38 | 6.34 | 12.63 |
| stall mio throttle | 0.53 | 0.60 | 1.58 | 2.13 |
| stall wait | 1.13 | 0.96 | 1.34 | 1.41 |
| warp latency / instruction | 8.91 | 6.37 | 25.97 | 36.93 |
| registers / thread | 128 | 119 | 128 | 80 |
| static smem B | 128 | 1664 | 1664 | 1728 |
