### forward

| metric | forward baseline | forward v3 T=1024 | forward T=768 |
|---|---:|---:|---:|
| time us | 412.6 | 277.1 | 253.5 |
| DRAM read MB | 31.9 | 39.6 | 35.1 |
| DRAM write MB | 17.7 | 15.1 | 18.6 |
| DRAM % peak | - | 44.2 | 47.4 |
| SM % peak | 33.2 | 53.3 | 52.6 |
| global ld sectors (k) | 6261 | 2214 | 2210 |
| global ld requests (k) | 1168 | 341 | 337 |
| L2 read hit % | 44.5 | 58.9 | 63.0 |
| L2 write hit % | 0.3 | 5.7 | 0.0 |
| shared wavefronts (k) | 10260 | 11988 | 9590 |
| instructions (M) | 55.63 | 59.04 | 53.33 |
| local ld instructions | 0 | 0 | 0 |
| warps active % | 33.2 | 65.3 | 49.7 |
| eligible warps / cycle | 0.68 | 2.26 | 1.40 |
| issue active % | 35.0 | 57.4 | 53.5 |
| stall long scoreboard | 4.41 | 1.26 | 1.43 |
| stall short scoreboard | 0.70 | 0.89 | 0.96 |
| stall barrier | 0.80 | 1.67 | 1.81 |
| stall mio throttle | 1.18 | 3.15 | 1.23 |
| stall wait | 1.61 | 1.73 | 1.68 |
| warp latency / instruction | 11.44 | 13.86 | 11.16 |
| registers / thread | 125 | 56 | 67 |
| static smem B | 0 | 1536 | 1536 |

### backward

| metric | backward baseline | backward v3 T=512 | backward T=512 |
|---|---:|---:|---:|
| time us | 864.8 | 574.3 | 585.4 |
| DRAM read MB | 83.8 | 59.3 | 63.9 |
| DRAM write MB | 57.2 | 55.9 | 52.9 |
| DRAM % peak | - | 44.8 | 44.6 |
| SM % peak | 42.2 | 59.8 | 59.8 |
| global ld sectors (k) | 10329 | 4223 | 4223 |
| global ld requests (k) | 1903 | 647 | 647 |
| L2 read hit % | 39.1 | 74.9 | 73.2 |
| L2 write hit % | 40.6 | 41.3 | 42.5 |
| shared wavefronts (k) | 15511 | 14569 | 14562 |
| instructions (M) | 147.51 | 136.55 | 136.55 |
| local ld instructions | 0 | 0 | 0 |
| warps active % | 33.4 | 33.3 | 33.0 |
| eligible warps / cycle | 0.92 | 1.73 | 1.74 |
| issue active % | 44.8 | 62.7 | 62.9 |
| stall long scoreboard | 3.52 | 0.40 | 0.39 |
| stall short scoreboard | 0.62 | 0.64 | 0.64 |
| stall barrier | 0.54 | 0.38 | 0.38 |
| stall mio throttle | 0.53 | 0.60 | 0.59 |
| stall wait | 1.13 | 0.96 | 0.96 |
| warp latency / instruction | 8.91 | 6.37 | 6.35 |
| registers / thread | 128 | 119 | 119 |
| static smem B | 128 | 1664 | 1664 |
