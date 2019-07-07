## Experimental Results

* see [data_analysis](data_analysis)
* Script-File-Organization see [FileOrg.md](FileOrg.md).  

## Dataset Directory Tree / Statistics

see [dataset_organization](dataset_organization) for more details. 

Dataset | V |  E | d | max-d | DODG-max-d 
--- | --- | --- | --- | --- | --- 
livejournal (LJ)  |  `4,036,538`  |  `34,681,189`  |  `17.2`  |  `14,815`  | `527` 
orkut (OR)   | `3,072,627`  |  `117,185,083`  |  `76.3`  |  `33,312` | `535` 
web-uk (WU)    | `18,520,343` | `150,568,277` | `16.3` | `169,382` | `943` 
web-eu (WE)   |  `11,264,052` | `191,414,391` | `34.0` | `183,591`  | `9,666` 
webbase (WB)  |  `118,142,143`  |  `525,013,368` |  `8.9`  |  `803,138` |   `1,225` 
web-it  (WI)  |  `41,291,083`  |  `583,044,292`  |  `28.2`  |  `1,243,927` | `3,212` 
twitter  (TW)  |  `41,652,230`  |  `684,500,375` |  `32.9`  |  `1,405,985`  | `3,317` 
friendster (FR)  |  `124,836,180`  |  `1,806,067,135` |  `28.9`  |  `5,214`  | `868`  
rmat_v0.5m_e0.5g |  `500,000` | `499,999,971` | `2000` | `4,936` | `192,655`
rmat_v5m_e0.5g  | `4,999,986` | `500,000,000` | `200` | `769` | `94,431`
rmat_v50m_e0.5g | `49,999,953` | `500,000,000 ` | `20` | `136` | `29,271`
rmat_v1m_e1g    | `1,000,000` | `999,999,986` | `2000` | `5,547` | `304,492` 
rmat_v10m_e1g   | `9,999,998` | `1,000,000,000` | `200` | `806` | `125,362`
rmat_v100m_e1g  | `99,999,795` | `1,000,000,000` | `20` | `141` | `41,197`
rmat_v2m_e2g | `1,999,997` | `1,999,999,989` | `2000` | `6,089` | `458,951`
rmat_v20m_e2g | `19,999,972` | `2,000,000,000` | `200` | `1,391` | `172,735`
rmat_v200m_e2g | `199,999,407` | `2000,000,000` | `20` | `155` | `53,954`

Dataset | DODG-max-d | TC | MC | CoreVal | TrussVal
--- | --- | --- | --- | --- | --- 
livejournal (LJ)  | `527` | `177,820,130` | `327` | `360` | `352`
orkut (OR)   | `535` |  `627,584,181` | `51` | `253` | `78`
web-uk (WU)  | `943` | `2,219,257,972` | `944` | `943` | `944`
web-eu (WE)  | `9,666` | `340,972,167,210` | `9,667` | `9,666` | `9,667` 
webbase (WB)  | `1,225` | `6,931,658,967` | `1,226` | `1,225` | `1,226`
web-it  (WI)  | `3,212` | `24,382,942,061` | `3,210` | `3,209` | `3,210`
twitter  (TW)  | `3,317` | `23,871,588,549` | `364` | `2,059` | `1,517`
friendster (FR) | `868`  | `4,173,724,142` | `129` | `304` | `129`
rmat_v0.5m_e0.5g |  `4,936` | `171,681,167,254` | `373` | `4,228` | `2,138`
rmat_v5m_e0.5g  | `769` | `1,705,342,424` | `30` | `627` | `99`
rmat_v50m_e0.5g | `136` | `6,127,241` | `5` | `84` | `5`
rmat_v1m_e1g   | `5,547` | `305,445,659,482` | `367` | `4,835` | `2,322`
rmat_v10m_e1g  | `806` | `2,288,408,709` | `19` | `679` | `86`
rmat_v100m_e1g  | `141` | `9,394,872` | `5` | `94` | `5`
rmat_v2m_e2g | `6,089` | `495,273,096,725` | `316` | `5,771` | `2,502`
rmat_v20m_e2g  | `1,391` | `3,341,193,251` | `19` | `773` | `78`
rmat_v200m_e2g | `155` | `12,665,950` | `4` | `104` | `5` 

Dataset | 4 | 5 | 6 | 7 | 8 | 9
--- | --- | --- | --- | --- | --- | --- 
LJ  | `5,216,918,441` |  ` 246,378,629,120` | `10,990,740,312,954`
OR   | `3,221,946,137` | ` 15,766,607,860` | `75,249,427,585` | `353,962,921,685` |  ` 1,632,691,821,296` | `7,248,102,160,867`
WU | `107,386,384,124` | `11,575,113,391,749` |
WE  | 
WB | `781,028,274,064` 
WI  | `7,510,704,698,598`
TW  | `3,629,832,195,348` | 
FR | `8,963,503,263` | `21,710,817,218` | ` 59,926,510,355` | 

## CUDA Experimental Results

### Core Checking Performance History

on the twiiter dataset (degreee high-skew)

algorithm | performance
--- | ---
avx512-merge | **613.63s** (KNL improved SCAN-XP merge-only after removing redudant computation)
galloping-single avx512/avx512-merge (KNL cache mode) | **58.07s** (our best strategy)
`edge_dst` allocated on MCDRAM (KNL flat mode) | **54.71s**
dense bitmap amortized construction, degree-descending (on CPU) | **41.88s**
unified memory, 2D bitmap with `BITMAP_SCALE_LOG = 9` degree-descending (on TitanXP single GPU) | **30.32s**, including **6s** binary-search on CPU, see [09-08-exp-res](python_experiments/data_analysis/data-md/lccpu12/09-08-cuda-large-bitmap-index-scale.md)
multi-GPU implementation with 2D bitmap (8 GPUs, LB by degree filtered accumulation) | **21.39s**, real `comp+transfer` time remvong task init time, on GPU is **8.4s**, see [09-10-exp-res](python_experiments/data_analysis/data-md/lccpu12/09-10-multi-gpu.md)
multi-GPU with less task-gen cost | **15.52s**, task-gen time beging **0.84s** from previous **3.8s**, reverting to `std::lower_bound` (AVX2 not useful, find the first `>=` pivot)
multi-GPU with co-processing on CPU (assigning reverse edge offset `(u,v)'s offset to common_node_num[e(v,u)]`) | **12.07s**, benefiting from the overlap
multi-GPU with dynamic load balance `8x` more tasks instead of `num-of-gpus` | **10.71s**, slightly balanced load at small cost, including CPU's `0.87s` symmetric assignment, `1.38s` task gen, `1.2s` sim-func-comp  