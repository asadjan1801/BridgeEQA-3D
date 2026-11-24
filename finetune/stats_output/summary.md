# Dataset Statistical Summary

Scenes scanned: **3220**

Total QA pairs: **63,079** (rated: 55,956, with image cite: 56,166)


## Scene size

|       |   n_nodes |    n_qa |   n_components |   n_elements |
|:------|----------:|--------:|---------------:|-------------:|
| count |   3220    | 3220    |        3220    |      3220    |
| mean  |     21.6  |   19.59 |           4.78 |         8.69 |
| std   |     12.59 |    3.16 |           1.66 |         4.04 |
| min   |      1    |    0    |           2    |         0    |
| 25%   |     12    |   20    |           4    |         6    |
| 50%   |     18    |   20    |           4    |         8    |
| 75%   |     27    |   20    |           5    |        10    |
| max   |    113    |   22    |          11    |        57    |


## Agency (normalized)

| agency         |   count |
|:---------------|--------:|
| State          |    1619 |
| Town/Township  |    1493 |
| City/Municipal |      79 |
| Village        |      23 |
| Railroad       |       5 |
| Private        |       1 |


## Inspection year

|   inspection_year |   count |
|------------------:|--------:|
|              2021 |      62 |
|              2022 |     172 |
|              2023 |    1131 |
|              2024 |    1433 |
|              2025 |     422 |


## Component ratings (0-9 scale, higher=better)

| component      |   count |   mean |   median |   std |
|:---------------|--------:|-------:|---------:|------:|
| Deck           |    4566 |   6.8  |        7 |  1.13 |
| Channel        |    3171 |   6.79 |        7 |  1    |
| Superstructure |    2537 |   6.81 |        7 |  1.02 |
| Substructure   |    2533 |   6.66 |        7 |  0.92 |
| Culverts       |     946 |   6.44 |        6 |  1.12 |
| Approach       |       9 |   7.78 |        8 |  0.67 |


## Element ratings top 15 (CS scale 1-4, lower=better)

| element                                 |   count |   mean |   median |
|:----------------------------------------|--------:|-------:|---------:|
| Reinforced Concrete Abutment            |    2363 |   1.68 |        1 |
| Reinforced Concrete Wing/Retaining Wall |    2170 |   1.65 |        1 |
| Metal Bridge Railing                    |    1667 |   1.45 |        1 |
| Concrete Fascia                         |    1587 |   1.58 |        1 |
| Wearing Surfaces                        |    1370 |   1.38 |        1 |
| Reinforced Concrete Deck                |    1342 |   1.73 |        2 |
| Steel Open Girder/Beam                  |    1310 |   1.75 |        2 |
| Cracking (RC and Other)                 |    1080 |   2.29 |        2 |
| Efflorescence/Rust Staining             |     833 |   2.24 |        2 |
| Delamination/Spall/Patched Area         |     624 |   2.96 |        3 |
| Corrosion                               |     598 |   2.66 |        2 |
| Movable Bearing                         |     568 |   1.98 |        2 |
| RC Slab                                 |     549 |   1.59 |        1 |
| Steel Protective Coating                |     524 |   2.28 |        2 |
| Fixed Bearing                           |     480 |   1.8  |        2 |


## QA-level

|       |   rating |   n_images_cited |   answer_len |
|:------|---------:|-----------------:|-------------:|
| count | 55956    |         63079    |     63079    |
| mean  |     6.6  |             1.8  |       286.32 |
| std   |     1.11 |             1.21 |       107.38 |
| min   |     1    |             0    |        14    |
| 25%   |     6    |             1    |       211    |
| 50%   |     7    |             2    |       272    |
| 75%   |     7    |             2    |       347    |
| max   |     9    |            20    |      1289    |


## Top 20 towns

| town        |   count |
|:------------|--------:|
| BENNINGTON  |      48 |
| STJOHNSBURY |      46 |
| CHESTER     |      46 |
| HARTFORD    |      42 |
| LYNDON      |      41 |
| BARNET      |      38 |
| SPRINGFIELD |      35 |
| BRATTLEBORO |      34 |
| BRIDGEWATER |      34 |
| WATERFORD   |      33 |
| STOWE       |      32 |
| PLYMOUTH    |      32 |
| BARTON      |      31 |
| SBURLINGTON |      30 |
| NORTHFIELD  |      29 |
| ROCHESTER   |      28 |
| WOODSTOCK   |      28 |
| RANDOLPH    |      27 |
| NEWBURY     |      27 |
| GUILFORD    |      26 |


## Plots

- ![component_rating_box](plots/component_rating_box.png)
- ![component_rating_heatmap](plots/component_rating_heatmap.png)
- ![component_score_distribution](plots/component_score_distribution.png)
- ![components_per_scene_hist](plots/components_per_scene_hist.png)
- ![element_rating_box](plots/element_rating_box.png)
- ![elements_per_scene_hist](plots/elements_per_scene_hist.png)
- ![frac_cited_hist](plots/frac_cited_hist.png)
- ![images_vs_rating](plots/images_vs_rating.png)
- ![qa_answer_len_hist](plots/qa_answer_len_hist.png)
- ![qa_images_cited_hist](plots/qa_images_cited_hist.png)
- ![qa_per_scene_hist](plots/qa_per_scene_hist.png)
- ![qa_rating_hist](plots/qa_rating_hist.png)
- ![scene_images_hist](plots/scene_images_hist.png)
- ![scenes_by_agency](plots/scenes_by_agency.png)
- ![scenes_by_year](plots/scenes_by_year.png)
- ![top_towns](plots/top_towns.png)