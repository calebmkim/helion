# T2 TRAIN 3-way bench: mine vs helion_default vs tc_default

ratio_hd = helion_default_us / mine_us  (>1 => MINE faster)
ratio_tc = tc_default_us / mine_us      (>1 => MINE faster)

## Per-kernel summary

### softmax  (n=15, gated_ok=15)
- vs helion_default: median ratio_hd=3.741 min=1.360 max=20.809 | mine wins>5%=15 tie=0 mine loses>5%=0
- vs tc_default:     median ratio_tc=1.079 min=0.883 max=1.433 | mine wins>5%=8 tie=5 mine loses>5%=2

### welford  (n=15, gated_ok=15)
- vs helion_default: median ratio_hd=2.611 min=1.964 max=5.290 | mine wins>5%=15 tie=0 mine loses>5%=0
- vs tc_default:     median ratio_tc=0.963 min=0.863 max=1.000 | mine wins>5%=0 tie=11 mine loses>5%=4

### kl_div  (n=13, gated_ok=13)
- vs helion_default: median ratio_hd=5.266 min=1.479 max=21.097 | mine wins>5%=13 tie=0 mine loses>5%=0
- vs tc_default:     median ratio_tc=1.051 min=1.040 max=1.369 | mine wins>5%=7 tie=6 mine loses>5%=0

### jsd  (n=13, gated_ok=13)
- vs helion_default: median ratio_hd=1.911 min=1.228 max=6.865 | mine wins>5%=13 tie=0 mine loses>5%=0
- vs tc_default:     median ratio_tc=1.013 min=0.994 max=1.030 | mine wins>5%=0 tie=13 mine loses>5%=0

## !! SHAPES WHERE MINE LOSES (ratio < 0.95) !!

- softmax(131072,256) LOSES to tc_default: ratio=0.883  mine_us=106.3 hd_us=146.7839926481247 tc_us=93.9 ratio_hd=1.3803791377012167 ratio_tc=0.8826361866613752
    mine_cfg={'block_sizes': [8, 256], 'num_warps': 4, 'num_stages': 1, 'pid_type': 'flat'}
    helion_default_cfg={'block_sizes': [32, 32], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'}
    mine_codegen=persistent  note=
- softmax(4096,16384) LOSES to tc_default: ratio=0.909  mine_us=204.4 hd_us=784.2559814453125 tc_us=185.7 ratio_hd=3.837169175978556 ratio_tc=0.9085643036172849
    mine_cfg={'block_sizes': [1, 16384], 'num_warps': 16, 'num_stages': 1, 'pid_type': 'flat'}
    helion_default_cfg={'block_sizes': [32, 32], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'}
    mine_codegen=persistent  note=
- welford(8192,12288) LOSES to tc_default: ratio=0.948  mine_us=291.9 hd_us=1023.472011089325 tc_us=276.8 ratio_hd=3.5063860661113955 ratio_tc=0.9481993102811723
    mine_cfg={'block_sizes': [1, 8192, 2048], 'load_eviction_policies': ['last', 'first', 'first', 'first'], 'num_warps': 16, 'num_stages': 1, 'pid_type': 'flat'}
    helion_default_cfg={'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 0], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'}
    mine_codegen=looped  note=
- welford(4096,16384) LOSES to tc_default: ratio=0.863  mine_us=214.1 hd_us=1132.5759887695312 tc_us=184.7 ratio_hd=5.290433284938899 ratio_tc=0.8626307592868102
    mine_cfg={'block_sizes': [1, 8192, 2048], 'load_eviction_policies': ['last', 'first', 'first', 'first'], 'num_warps': 16, 'num_stages': 1, 'pid_type': 'flat'}
    helion_default_cfg={'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 0], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'}
    mine_codegen=looped  note=
- welford(65536,4096) LOSES to tc_default: ratio=0.915  mine_us=772.6 hd_us=1639.0399932861328 tc_us=706.7 ratio_hd=2.1214380787351788 ratio_tc=0.9146993132575916
    mine_cfg={'block_sizes': [4, 4096, 2048], 'load_eviction_policies': ['last', 'first', 'first', 'first'], 'num_warps': 8, 'num_stages': 1, 'pid_type': 'flat'}
    helion_default_cfg={'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 0], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'}
    mine_codegen=looped  note=
- welford(32768,8192) LOSES to tc_default: ratio=0.906  mine_us=778.0 hd_us=1689.5359754562378 tc_us=705.0 ratio_hd=2.1715061555620534 ratio_tc=0.9061034993046682
    mine_cfg={'block_sizes': [2, 8192, 2048], 'load_eviction_policies': ['last', 'first', 'first', 'first'], 'num_warps': 16, 'num_stages': 1, 'pid_type': 'flat'}
    helion_default_cfg={'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 0], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'}
    mine_codegen=looped  note=

## Per-shape table

kernel | M | N | mine_us | hd_us | tc_us | ratio_hd | ratio_tc | mine_ok | hd_ok | codegen | note
---|---|---|---|---|---|---|---|---|---|---|---
softmax | 262144 | 128 | 99.8 | 135.8 | 94.9 | 1.360 | 0.951 | True | True | persistent | 
softmax | 131072 | 256 | 106.3 | 146.8 | 93.9 | 1.380 | 0.883 | True | True | persistent | 
softmax | 16384 | 512 | 29.2 | 49.2 | 27.8 | 1.685 | 0.952 | True | True | persistent | 
softmax | 8192 | 1024 | 29.2 | 65.1 | 28.3 | 2.231 | 0.968 | True | True | persistent | 
softmax | 8192 | 2048 | 51.4 | 127.5 | 55.9 | 2.478 | 1.087 | True | True | persistent | 
softmax | 8192 | 2560 | 63.2 | 154.6 | 90.6 | 2.446 | 1.433 | True | True | persistent | 
softmax | 4096 | 3072 | 41.0 | 153.4 | 50.5 | 3.740 | 1.231 | True | True | persistent | 
softmax | 4096 | 4096 | 52.7 | 197.3 | 51.9 | 3.741 | 0.985 | True | True | persistent | 
softmax | 4096 | 5120 | 63.4 | 249.5 | 71.8 | 3.938 | 1.134 | True | True | persistent | 
softmax | 4096 | 8192 | 98.1 | 388.8 | 96.5 | 3.962 | 0.983 | True | True | persistent | 
softmax | 4096 | 16384 | 204.4 | 784.3 | 185.7 | 3.837 | 0.909 | True | True | persistent | 
softmax | 2048 | 24576 | 144.3 | 1054.1 | 184.8 | 7.307 | 1.281 | True | True | persistent | 
softmax | 2048 | 32768 | 195.9 | 1416.6 | 238.8 | 7.231 | 1.219 | True | True | persistent | 
softmax | 1024 | 65536 | 202.3 | 2739.4 | 270.3 | 13.543 | 1.336 | True | True | looped | 
softmax | 512 | 131072 | 256.3 | 5333.0 | 276.5 | 20.809 | 1.079 | True | True | looped | 
welford | 16384 | 768 | 40.5 | 79.6 | 39.7 | 1.964 | 0.980 | True | True | persistent | 
welford | 16384 | 1024 | 50.8 | 104.6 | 50.7 | 2.057 | 0.998 | True | True | persistent | 
welford | 16384 | 1536 | 76.1 | 189.2 | 72.5 | 2.485 | 0.953 | True | True | persistent | 
welford | 16384 | 2048 | 94.7 | 261.4 | 94.7 | 2.760 | 1.000 | True | True | persistent | 
welford | 16384 | 2560 | 122.2 | 309.6 | 117.6 | 2.534 | 0.962 | True | True | persistent | 
welford | 8192 | 3072 | 74.8 | 261.6 | 72.9 | 3.497 | 0.975 | True | True | persistent | 
welford | 16384 | 4096 | 185.3 | 483.7 | 182.3 | 2.611 | 0.984 | True | True | looped | 
welford | 8192 | 5120 | 121.9 | 430.0 | 116.4 | 3.529 | 0.955 | True | True | looped | 
welford | 8192 | 7168 | 167.1 | 592.0 | 160.9 | 3.544 | 0.963 | True | True | looped | 
welford | 8192 | 8192 | 188.0 | 657.8 | 182.5 | 3.500 | 0.971 | True | True | looped | 
welford | 8192 | 12288 | 291.9 | 1023.5 | 276.8 | 3.506 | 0.948 | True | True | looped | 
welford | 4096 | 16384 | 214.1 | 1132.6 | 184.7 | 5.290 | 0.863 | True | True | looped | 
welford | 262144 | 2048 | 1432.8 | 3158.4 | 1405.9 | 2.204 | 0.981 | True | True | persistent | 
welford | 65536 | 4096 | 772.6 | 1639.0 | 706.7 | 2.121 | 0.915 | True | True | looped | 
welford | 32768 | 8192 | 778.0 | 1689.5 | 705.0 | 2.172 | 0.906 | True | True | looped | 
kl_div | 8192 | 30522 | 652.5 | 964.9 | 726.8 | 1.479 | 1.114 | True | True | looped | 
kl_div | 8192 | 32000 | 677.1 | 1895.9 | 715.9 | 2.800 | 1.057 | True | True | looped | 
kl_div | 4096 | 32064 | 353.2 | 1825.8 | 483.4 | 5.169 | 1.369 | True | True | looped | 
kl_div | 8192 | 50257 | 1088.4 | 1691.0 | 1152.0 | 1.554 | 1.058 | True | True | looped | 
kl_div | 4096 | 50304 | 542.0 | 2853.9 | 566.2 | 5.266 | 1.045 | True | True | looped | 
kl_div | 4096 | 65536 | 695.1 | 2573.5 | 728.4 | 3.702 | 1.048 | True | True | looped | 
kl_div | 4096 | 49152 | 527.6 | 2782.9 | 558.8 | 5.274 | 1.059 | True | True | looped | 
kl_div | 2048 | 98304 | 529.2 | 5519.9 | 556.3 | 10.430 | 1.051 | True | True | looped | 
kl_div | 4096 | 128256 | 1338.9 | 7232.0 | 1391.9 | 5.401 | 1.040 | True | True | looped | 
kl_div | 2048 | 128000 | 682.9 | 7168.3 | 716.1 | 10.497 | 1.049 | True | True | looped | 
kl_div | 2048 | 151936 | 806.3 | 8405.3 | 865.5 | 10.425 | 1.073 | True | True | looped | 
kl_div | 1024 | 256000 | 683.6 | 14421.8 | 716.2 | 21.097 | 1.048 | True | True | looped | 
kl_div | 16384 | 32000 | 1325.3 | 2061.9 | 1391.1 | 1.556 | 1.050 | True | True | looped | 
jsd | 8192 | 30522 | 676.0 | 1268.2 | 680.9 | 1.876 | 1.007 | True | True | looped | 
jsd | 8192 | 32000 | 700.2 | 1292.0 | 705.1 | 1.845 | 1.007 | True | True | looped | 
jsd | 8192 | 32064 | 700.1 | 1291.2 | 696.0 | 1.844 | 0.994 | True | True | looped | 
jsd | 8192 | 50257 | 1150.3 | 2196.0 | 1184.6 | 1.909 | 1.030 | True | True | looped | 
jsd | 8192 | 50304 | 1073.6 | 2022.6 | 1087.3 | 1.884 | 1.013 | True | True | looped | 
jsd | 8192 | 65536 | 1371.3 | 2678.1 | 1397.5 | 1.953 | 1.019 | True | True | looped | 
jsd | 4096 | 49152 | 538.9 | 1823.3 | 548.3 | 3.383 | 1.017 | True | True | looped | 
jsd | 4096 | 98304 | 1047.9 | 3626.8 | 1064.4 | 3.461 | 1.016 | True | True | looped | 
jsd | 8192 | 128256 | 2675.1 | 5112.8 | 2685.9 | 1.911 | 1.004 | True | True | looped | 
jsd | 4096 | 128000 | 1366.1 | 4734.5 | 1386.7 | 3.466 | 1.015 | True | True | looped | 
jsd | 4096 | 151936 | 1619.6 | 5603.3 | 1638.1 | 3.460 | 1.011 | True | True | looped | 
jsd | 2048 | 256000 | 1354.8 | 9300.7 | 1390.8 | 6.865 | 1.027 | True | True | looped | 
jsd | 16384 | 32000 | 1372.2 | 1684.4 | 1369.8 | 1.228 | 0.998 | True | True | looped | 
