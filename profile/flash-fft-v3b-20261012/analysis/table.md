### forward

| metric | forward baseline | forward v3a T=1024 | forward T=512 | forward T=1024 |
|---|---:|---:|---:|---:|
| time us | 412.6 | 277.3 | 285.5 | 299.1 |
| DRAM read MB | 31.9 | 42.1 | 34.4 | 39.3 |
| DRAM write MB | 17.7 | 18.0 | 15.6 | 11.4 |
| DRAM % peak | - | 48.4 | 39.1 | 37.9 |
| SM % peak | 33.2 | 53.9 | 48.2 | 52.7 |
| global ld sectors (k) | 6261 | 2214 | 2206 | 2214 |
| global ld requests (k) | 1168 | 341 | 333 | 341 |
| L2 read hit % | 44.5 | 57.0 | 63.6 | 59.1 |
| L2 write hit % | 0.3 | 29.9 | 21.3 | 25.4 |
| shared wavefronts (k) | 10260 | 11988 | 9922 | 14902 |
| instructions (M) | 55.63 | 59.04 | 54.79 | 62.37 |
| local ld instructions | 0 | 0 | 0 | 0 |
| warps active % | 33.2 | 63.6 | 32.5 | 65.5 |
| eligible warps / cycle | 0.68 | 2.24 | 1.03 | 2.22 |
| issue active % | 35.0 | 56.6 | 51.1 | 56.1 |
| stall long scoreboard | 4.41 | 1.13 | 0.96 | 0.85 |
| stall short scoreboard | 0.70 | 0.89 | 0.87 | 1.10 |
| stall barrier | 0.80 | 1.67 | 0.44 | 1.91 |
| stall mio throttle | 1.18 | 3.15 | 0.91 | 3.33 |
| stall wait | 1.61 | 1.73 | 1.65 | 1.72 |
| warp latency / instruction | 11.44 | 14.05 | 7.81 | 14.19 |
| registers / thread | 125 | 56 | 64 | 59 |
| static smem B | 0 | 1536 | 1536 | 1536 |

### backward

| metric | backward baseline | backward v3a T=1024 | backward v3a T=512 | backward T=512 | backward T=1024 |
|---|---:|---:|---:|---:|---:|
| time us | 864.8 | 617.8 | 613.5 | 578.3 | 664.1 |
| DRAM read MB | 83.8 | 60.6 | 64.1 | 59.9 | 67.5 |
| DRAM write MB | 57.2 | 61.3 | 59.0 | 49.4 | 67.3 |
| DRAM % peak | - | 44.1 | 44.9 | 42.2 | 45.3 |
| SM % peak | 42.2 | 62.0 | 57.5 | 61.5 | 60.0 |
| global ld sectors (k) | 10329 | 4231 | 4223 | 4223 | 4231 |
| global ld requests (k) | 1903 | 655 | 647 | 647 | 655 |
| L2 read hit % | 39.1 | 77.5 | 72.8 | 75.0 | 78.1 |
| L2 write hit % | 40.6 | 41.5 | 42.5 | 42.1 | 45.0 |
| shared wavefronts (k) | 15511 | 18452 | 17798 | 15189 | 23406 |
| instructions (M) | 147.51 | 148.05 | 141.50 | 142.59 | 154.80 |
| local ld instructions | 0 | 32256 | 0 | 0 | 327168 |
| warps active % | 33.4 | 66.0 | 33.4 | 36.7 | 66.7 |
| eligible warps / cycle | 0.92 | 3.51 | 1.60 | 1.73 | 3.21 |
| issue active % | 44.8 | 65.7 | 60.7 | 64.7 | 63.2 |
| stall long scoreboard | 3.52 | 0.58 | 0.40 | 0.93 | 0.59 |
| stall short scoreboard | 0.62 | 0.98 | 0.78 | 0.53 | 1.32 |
| stall barrier | 0.54 | 1.31 | 0.46 | 0.36 | 1.50 |
| stall mio throttle | 0.53 | 1.70 | 0.70 | 0.59 | 1.84 |
| stall wait | 1.13 | 1.12 | 1.01 | 1.01 | 1.17 |
| warp latency / instruction | 8.91 | 12.16 | 6.58 | 6.17 | 12.64 |
| registers / thread | 128 | 64 | 125 | 119 | 64 |
| static smem B | 128 | 1792 | 1664 | 1664 | 1792 |
