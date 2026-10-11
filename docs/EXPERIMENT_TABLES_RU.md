# Полные таблицы экспериментов

Автоматическое приложение к [отчёту](../REPORT_RU.md). Solved приведён в процентах. Основная серия: test; пилоты: validation; overfit: train_subset. Пауза LR-кандидата означает завершение его отборочного бюджета, а не финальное обучение.

## 60 финальных запусков

| Задача | Метод | Seed | Solved % | Loss | Best step | Train s | Config hash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| maze | onehop | 11 | 0.700000 | 0.6150305943489075 | 5500 | 285.159 | 179a844d22ca2697 |
| maze | onehop | 23 | 0.600000 | 0.6243695950508118 | 2500 | 283.133 | c9bcb6f9f20f9d1e |
| maze | onehop | 37 | 0.600000 | 0.6229411859512329 | 2500 | 280.848 | 6deb324d301ec5e9 |
| maze | onehop | 53 | 0.600000 | 0.618036084651947 | 4250 | 283.115 | d9ca92d3d6fc4d8f |
| maze | onehop | 71 | 0.600000 | 0.6236941618919373 | 3000 | 281.535 | e32fc0b4b03ddd8d |
| maze | binomial | 11 | 99.000000 | 0.004018487564695534 | 16500 | 597.731 | 15751027a78eefc8 |
| maze | binomial | 23 | 98.900000 | 0.009576574480239287 | 17250 | 592.633 | d9660122458ed0c0 |
| maze | binomial | 37 | 98.700000 | 0.00450705348898191 | 14000 | 593.630 | fcff4ff7f9f365a2 |
| maze | binomial | 53 | 98.600000 | 0.01692495057155611 | 17000 | 591.961 | b4227da405c21eaf |
| maze | binomial | 71 | 98.600000 | 0.005964416787726805 | 12250 | 591.532 | 0cf228f2a68c50e7 |
| maze | exponential_truncated | 11 | 99.500000 | 0.0015780523340399668 | 12000 | 599.494 | ba9a181d6ca9d7e2 |
| maze | exponential_truncated | 23 | 99.800000 | 0.00039503516320473864 | 18500 | 592.279 | 02db194b977dbe11 |
| maze | exponential_truncated | 37 | 99.800000 | 0.0002064029014418338 | 14250 | 592.710 | 7652c9fd862e643f |
| maze | exponential_truncated | 53 | 99.500000 | 0.0005304988019634038 | 13000 | 593.316 | 9d70e876070dc7e9 |
| maze | exponential_truncated | 71 | 99.300000 | 0.0031034069046436343 | 18000 | 592.115 | 5e20af0d98383256 |
| maze | finite_geometric | 11 | 99.900000 | 0.002169356179458177 | 25750 | 594.288 | 6f9672c7471c9bd9 |
| maze | finite_geometric | 23 | 99.900000 | 0.007628172563392582 | 13250 | 592.885 | 52bfceace577d4a8 |
| maze | finite_geometric | 37 | 99.900000 | 9.266648621952101e-05 | 28250 | 593.091 | f31e328f9d1b25a1 |
| maze | finite_geometric | 53 | 99.900000 | 0.00028446787635039075 | 13000 | 591.078 | 023d95ece6754fba |
| maze | finite_geometric | 71 | 99.800000 | 0.00043997194664552807 | 16750 | 591.492 | 32657aac5b0be9b9 |
| maze | learned_polynomial | 11 | 99.700000 | 0.00043984854950031147 | 20750 | 651.479 | dad9f315b261e310 |
| maze | learned_polynomial | 23 | 100.000000 | 3.2126814876392016e-06 | 13750 | 651.077 | 4fb0c3c1ae6aab2a |
| maze | learned_polynomial | 37 | 100.000000 | 4.39103099552085e-08 | 25750 | 651.624 | 2bab6e2cc3d69719 |
| maze | learned_polynomial | 53 | 99.800000 | 0.0001433571206249269 | 22250 | 651.382 | 5893ad40b287b946 |
| maze | learned_polynomial | 71 | 99.900000 | 0.00023954667369253003 | 9250 | 651.103 | ed8581ccfc61ae7a |
| maze | recurrent | 11 | 98.900000 | 0.0034195154160261156 | 30000 | 1515.842 | 99313afb106ececf |
| maze | recurrent | 23 | 99.500000 | 0.0043955343596171585 | 27000 | 1523.809 | 4c2c62498ef96c93 |
| maze | recurrent | 37 | 84.000000 | 0.05287079963088036 | 28750 | 1516.268 | ba83e3c3078f7bcc |
| maze | recurrent | 53 | 99.600000 | 0.000718243243987672 | 29500 | 1524.277 | 4e3c2474d5df288b |
| maze | recurrent | 71 | 100.000000 | 1.0350978081987705e-05 | 30000 | 1523.410 | 9dd50f5a039f9cf8 |
| sudoku | onehop | 11 | 6.061111 | 0.44140101952022975 | 26000 | 288.434 | cb56bf2f0b9a5dda |
| sudoku | onehop | 23 | 5.238889 | 0.4588263385825687 | 25250 | 289.731 | 4eea932a3fc2ca5c |
| sudoku | onehop | 37 | 5.322222 | 0.45789940796958073 | 25750 | 288.382 | 37e66cb902290a9b |
| sudoku | onehop | 53 | 5.344444 | 0.45479906537797715 | 27500 | 289.650 | 0e106a23141b32ac |
| sudoku | onehop | 71 | 5.500000 | 0.4508621243370904 | 29000 | 287.357 | 74fcf77235af189a |
| sudoku | binomial | 11 | 6.055556 | 0.4437653922504849 | 28250 | 353.075 | 8b215625b6d86ebe |
| sudoku | binomial | 23 | 4.588889 | 0.4797259228494432 | 27250 | 353.164 | b71e3c93ab33aa48 |
| sudoku | binomial | 37 | 4.083333 | 0.47578201497925654 | 29750 | 348.674 | df2ad4acea2673e4 |
| sudoku | binomial | 53 | 6.655556 | 0.4453598650826348 | 29250 | 352.770 | e34fdbc3dfe15af8 |
| sudoku | binomial | 71 | 4.755556 | 0.4611695850690206 | 30000 | 351.390 | 70541bc0c4001af3 |
| sudoku | exponential_truncated | 11 | 6.638889 | 0.4381473086145189 | 29500 | 353.975 | 2a41f2412a978f26 |
| sudoku | exponential_truncated | 23 | 5.611111 | 0.4565915018982357 | 29250 | 354.081 | 787c5025b9169835 |
| sudoku | exponential_truncated | 37 | 4.211111 | 0.4778453449408213 | 29000 | 351.272 | 86d3731fcf772063 |
| sudoku | exponential_truncated | 53 | 6.238889 | 0.4468238142066532 | 28500 | 353.555 | 8dcb0ad5d6840ce8 |
| sudoku | exponential_truncated | 71 | 6.111111 | 0.44702422926161023 | 30000 | 351.577 | 78c2e461dd792b4c |
| sudoku | finite_geometric | 11 | 5.938889 | 0.4547952262560527 | 29000 | 352.790 | b3cd46a9cefaf3a2 |
| sudoku | finite_geometric | 23 | 4.783333 | 0.47088308535681833 | 28500 | 352.641 | 85bf7a028d72a9c4 |
| sudoku | finite_geometric | 37 | 4.444444 | 0.4760028338432312 | 29750 | 353.285 | f40cfdcf5aad3562 |
| sudoku | finite_geometric | 53 | 5.016667 | 0.45923802134725783 | 28750 | 350.433 | 9de16f6cdc475f50 |
| sudoku | finite_geometric | 71 | 6.511111 | 0.4463236031002469 | 28750 | 355.151 | 4b97cb3f6b33f9e0 |
| sudoku | learned_polynomial | 11 | 5.861111 | 0.438301455338796 | 27000 | 377.708 | a03eea1d705e71cf |
| sudoku | learned_polynomial | 23 | 6.394444 | 0.4389460120201111 | 29000 | 377.591 | ace495b2dc92f317 |
| sudoku | learned_polynomial | 37 | 5.488889 | 0.45097510062323676 | 25750 | 377.473 | 760e34459c8b76fa |
| sudoku | learned_polynomial | 53 | 6.577778 | 0.4334029932551914 | 28750 | 374.840 | 4c5de467d80b5bd5 |
| sudoku | learned_polynomial | 71 | 5.905556 | 0.43659808667500816 | 27000 | 380.309 | 6b86284f07ae607c |
| sudoku | recurrent | 11 | 94.927778 | 0.03495615258815491 | 28750 | 1540.474 | a87129633ce3bbd2 |
| sudoku | recurrent | 23 | 94.577778 | 0.03746835642280833 | 29500 | 1538.735 | a6fe8e2ad0cc7a80 |
| sudoku | recurrent | 37 | 95.338889 | 0.031730386669036104 | 28750 | 1539.602 | 2c8cf3d0b60b2285 |
| sudoku | recurrent | 53 | 94.450000 | 0.039467440060740856 | 28250 | 1531.592 | a87b87f26f1fa7ec |
| sudoku | recurrent | 71 | 94.544444 | 0.0394696082319424 | 29750 | 1568.885 | ea08dc08c9dcd08e |

## mlp_pilot

| Задача | Метод | Seed | H | β | LR | Updates | Status | Split | Solved % | Config hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maze | binomial | 101 | 16 | 0.0 | 0.0003 | 500 | completed | validation | 2.400000 | 66c769515f9d0434 |
| maze | binomial | 101 | 16 | 0.1 | 0.0003 | 500 | completed | validation | 4.400000 | 84aa1d2fc0efa772 |
| maze | binomial | 103 | 16 | 0.0 | 0.0003 | 500 | completed | validation | 1.500000 | ffad6507501da005 |
| maze | binomial | 103 | 16 | 0.1 | 0.0003 | 500 | completed | validation | 2.800000 | b607a46f2f4cae90 |
| maze | exponential_truncated | 101 | 16 | 0.0 | 0.001 | 500 | completed | validation | 55.100000 | 48dc43c58abdf547 |
| maze | exponential_truncated | 101 | 16 | 0.1 | 0.001 | 500 | completed | validation | 71.200000 | 919abb14a8e7c186 |
| maze | exponential_truncated | 103 | 16 | 0.0 | 0.001 | 500 | completed | validation | 71.100000 | 1b9e588f62d87127 |
| maze | exponential_truncated | 103 | 16 | 0.1 | 0.001 | 500 | completed | validation | 78.100000 | 88c26f00179d0cde |
| maze | finite_geometric | 101 | 16 | 0.0 | 0.001 | 500 | completed | validation | 53.200000 | e04ef4ccf3f741f6 |
| maze | finite_geometric | 101 | 16 | 0.1 | 0.001 | 500 | completed | validation | 76.900000 | 29bc2903adb275e6 |
| maze | finite_geometric | 103 | 16 | 0.0 | 0.001 | 500 | completed | validation | 79.800000 | 4722a294fe4115a6 |
| maze | finite_geometric | 103 | 16 | 0.1 | 0.001 | 500 | completed | validation | 86.300000 | 088e131314ab7815 |
| sudoku | binomial | 101 | 4 | 0.0 | 0.001 | 500 | completed | validation | 0.055556 | 224c2b3fea6b2a88 |
| sudoku | binomial | 101 | 4 | 0.1 | 0.001 | 500 | completed | validation | 0.044444 | fcfc90dc810c7df8 |
| sudoku | binomial | 103 | 4 | 0.0 | 0.001 | 500 | completed | validation | 0.094444 | fe3116955fc65b31 |
| sudoku | binomial | 103 | 4 | 0.1 | 0.001 | 500 | completed | validation | 0.122222 | 00c5c531da5acc75 |
| sudoku | exponential_truncated | 101 | 4 | 0.0 | 0.001 | 500 | completed | validation | 0.038889 | d6bac7dfcd35bf4d |
| sudoku | exponential_truncated | 101 | 4 | 0.1 | 0.001 | 500 | completed | validation | 0.050000 | 4f49074b04cc6cd7 |
| sudoku | exponential_truncated | 103 | 4 | 0.0 | 0.001 | 500 | completed | validation | 0.111111 | 62347b2b68d0201e |
| sudoku | exponential_truncated | 103 | 4 | 0.1 | 0.001 | 500 | completed | validation | 0.133333 | 2b5d35ebd7fe723a |
| sudoku | finite_geometric | 101 | 4 | 0.0 | 0.001 | 500 | completed | validation | 0.072222 | 608c311a491cbb12 |
| sudoku | finite_geometric | 101 | 4 | 0.1 | 0.001 | 500 | completed | validation | 0.111111 | 1ebea083ebea17a0 |
| sudoku | finite_geometric | 103 | 4 | 0.0 | 0.001 | 500 | completed | validation | 0.083333 | 4a73a8a2c54351a0 |
| sudoku | finite_geometric | 103 | 4 | 0.1 | 0.001 | 500 | completed | validation | 0.138889 | a5ed81235a393905 |

## overfit

| Задача | Метод | Seed | H | β | LR | Updates | Status | Split | Solved % | Config hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maze | binomial | 101 | 8 | 0.0 | 0.001 | 1000 | completed | train_subset | 100.000000 | 9bb63f7202b717f8 |
| maze | exponential_truncated | 101 | 8 | 0.0 | 0.001 | 1000 | completed | train_subset | 100.000000 | 30f330df88eb7455 |
| maze | finite_geometric | 101 | 8 | 0.0 | 0.001 | 1000 | completed | train_subset | 100.000000 | 8f3aeb641d19eac2 |
| maze | learned_polynomial | 101 | 8 | 0.0 | 0.001 | 1000 | completed | train_subset | 100.000000 | ae74b07afddc4c7f |
| maze | onehop | 101 | 1 | 0.0 | 0.001 | 1000 | completed | train_subset | 82.812500 | dfe44f8e8a62ef29 |
| maze | recurrent | 101 | 1 | 0.0 | 0.001 | 1000 | completed | train_subset | 96.875000 | a5a45a76ba788df2 |
| sudoku | binomial | 101 | 8 | 0.0 | 0.001 | 1000 | completed | train_subset | 100.000000 | 8bafd4eac0ee5e45 |
| sudoku | exponential_truncated | 101 | 8 | 0.0 | 0.001 | 1000 | completed | train_subset | 100.000000 | b833a8f478638abf |
| sudoku | finite_geometric | 101 | 8 | 0.0 | 0.001 | 1000 | completed | train_subset | 100.000000 | ab7f7af9d524bd87 |
| sudoku | learned_polynomial | 101 | 8 | 0.0 | 0.001 | 1000 | completed | train_subset | 100.000000 | c8ada82443d211dd |
| sudoku | onehop | 101 | 1 | 0.0 | 0.001 | 1000 | completed | train_subset | 100.000000 | dd15f4a860f27f9a |
| sudoku | recurrent | 101 | 1 | 0.0 | 0.001 | 1000 | completed | train_subset | 100.000000 | 377d98ffd8442e8e |

## pilot

| Задача | Метод | Seed | H | β | LR | Updates | Status | Split | Solved % | Config hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maze | binomial | 101 | 8 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | 4d9832b368b5d35a |
| maze | binomial | 101 | 8 | 0.0 | 0.0003 | 2000 | completed | validation | 13.600000 | bc460f71a0f059aa |
| maze | binomial | 101 | 8 | 0.0 | 0.001 | 500 | paused | validation | 1.400000 | abbe5ccee8d83ad3 |
| maze | exponential_truncated | 101 | 8 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | 066d9439d31b953a |
| maze | exponential_truncated | 101 | 8 | 0.0 | 0.0003 | 500 | paused | validation | 1.700000 | e318f848b3e1a39c |
| maze | exponential_truncated | 101 | 8 | 0.0 | 0.001 | 2000 | completed | validation | 49.200000 | a04ebc7797f71540 |
| maze | finite_geometric | 101 | 8 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | 975c8ab21c6f9a53 |
| maze | finite_geometric | 101 | 8 | 0.0 | 0.0003 | 500 | paused | validation | 0.400000 | 9c1e3eb0d9c37f1b |
| maze | finite_geometric | 101 | 8 | 0.0 | 0.001 | 2000 | completed | validation | 64.000000 | 05ed334787047478 |
| maze | learned_polynomial | 101 | 8 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | 66662428c71d6a0b |
| maze | learned_polynomial | 101 | 8 | 0.0 | 0.0003 | 500 | paused | validation | 1.900000 | 0b3f9198635b005d |
| maze | learned_polynomial | 101 | 8 | 0.0 | 0.001 | 2000 | completed | validation | 53.400000 | e7e67f262edc30fd |
| maze | onehop | 101 | 1 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | 712be5c35fdbacc9 |
| maze | onehop | 101 | 1 | 0.0 | 0.0003 | 500 | paused | validation | 0.000000 | 1b9f32bff7bff260 |
| maze | onehop | 101 | 1 | 0.0 | 0.001 | 2000 | completed | validation | 0.200000 | eaafa2dec5c86321 |
| maze | recurrent | 101 | 1 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | 081faa66b675d815 |
| maze | recurrent | 101 | 1 | 0.0 | 0.0003 | 500 | paused | validation | 0.100000 | 33818c76238375eb |
| maze | recurrent | 101 | 1 | 0.0 | 0.001 | 2000 | completed | validation | 54.500000 | 8094cb147b6740d5 |
| sudoku | binomial | 101 | 8 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | 149374d2c89111c9 |
| sudoku | binomial | 101 | 8 | 0.0 | 0.0003 | 500 | paused | validation | 0.000000 | b27013ff9787a7e6 |
| sudoku | binomial | 101 | 8 | 0.0 | 0.001 | 2000 | completed | validation | 0.055556 | a6216b1c4e69bbca |
| sudoku | exponential_truncated | 101 | 8 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | cef024feabcf99b4 |
| sudoku | exponential_truncated | 101 | 8 | 0.0 | 0.0003 | 500 | paused | validation | 0.000000 | 7aaead6784074533 |
| sudoku | exponential_truncated | 101 | 8 | 0.0 | 0.001 | 2000 | completed | validation | 0.061111 | 7dd10eab8a8825d6 |
| sudoku | finite_geometric | 101 | 8 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | cf9cdbb0d3bd6dac |
| sudoku | finite_geometric | 101 | 8 | 0.0 | 0.0003 | 500 | paused | validation | 0.000000 | 6106de89fe26171b |
| sudoku | finite_geometric | 101 | 8 | 0.0 | 0.001 | 2000 | completed | validation | 0.261111 | ae26b81119626150 |
| sudoku | learned_polynomial | 101 | 8 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | fb6570f8604d41e9 |
| sudoku | learned_polynomial | 101 | 8 | 0.0 | 0.0003 | 500 | paused | validation | 0.000000 | 2b47beb15f7dfc76 |
| sudoku | learned_polynomial | 101 | 8 | 0.0 | 0.001 | 2000 | completed | validation | 0.244444 | d920319a8aabfd29 |
| sudoku | onehop | 101 | 1 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | b1276badf52b00da |
| sudoku | onehop | 101 | 1 | 0.0 | 0.0003 | 500 | paused | validation | 0.005556 | 6295def1a168ed04 |
| sudoku | onehop | 101 | 1 | 0.0 | 0.001 | 2000 | completed | validation | 1.794444 | 808379e923b47d38 |
| sudoku | recurrent | 101 | 1 | 0.0 | 0.0001 | 500 | paused | validation | 0.000000 | 486e0a98ba42ecc8 |
| sudoku | recurrent | 101 | 1 | 0.0 | 0.0003 | 500 | paused | validation | 10.255556 | 9fd991a9ab6c225d |
| sudoku | recurrent | 101 | 1 | 0.0 | 0.001 | 2000 | completed | validation | 63.305556 | d4c78809c1b018f3 |

## horizon_pilot

| Задача | Метод | Seed | H | β | LR | Updates | Status | Split | Solved % | Config hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maze | binomial | 101 | 16 | 0.0 | 0.0003 | 1000 | completed | validation | 37.700000 | 1596aa7d3401a987 |
| maze | binomial | 101 | 4 | 0.0 | 0.0003 | 1000 | completed | validation | 0.700000 | def3a13e37f6eb47 |
| maze | binomial | 101 | 8 | 0.0 | 0.0003 | 1000 | completed | validation | 5.000000 | 0799eb925b3e69e8 |
| maze | exponential_truncated | 101 | 16 | 0.0 | 0.001 | 1000 | completed | validation | 77.400000 | 36e1149ebc9eb9bd |
| maze | exponential_truncated | 101 | 4 | 0.0 | 0.001 | 1000 | completed | validation | 2.600000 | 08beca8b141d4ced |
| maze | exponential_truncated | 101 | 8 | 0.0 | 0.001 | 1000 | completed | validation | 18.400000 | 7a90f5f500a159d8 |
| maze | finite_geometric | 101 | 16 | 0.0 | 0.001 | 1000 | completed | validation | 87.200000 | 077381b0b0413259 |
| maze | finite_geometric | 101 | 4 | 0.0 | 0.001 | 1000 | completed | validation | 2.000000 | 9ef5f0de2597b6a3 |
| maze | finite_geometric | 101 | 8 | 0.0 | 0.001 | 1000 | completed | validation | 21.800000 | 67163fd56b335d36 |
| sudoku | binomial | 101 | 16 | 0.0 | 0.001 | 1000 | completed | validation | 0.000000 | 533e728ff2fa4cd0 |
| sudoku | binomial | 101 | 4 | 0.0 | 0.001 | 1000 | completed | validation | 0.127778 | a80d4eb7d0566ff3 |
| sudoku | binomial | 101 | 8 | 0.0 | 0.001 | 1000 | completed | validation | 0.033333 | 160d79ed159940d4 |
| sudoku | exponential_truncated | 101 | 16 | 0.0 | 0.001 | 1000 | completed | validation | 0.005556 | 0c77ab7f00fe8a01 |
| sudoku | exponential_truncated | 101 | 4 | 0.0 | 0.001 | 1000 | completed | validation | 0.033333 | b718274d58678b9a |
| sudoku | exponential_truncated | 101 | 8 | 0.0 | 0.001 | 1000 | completed | validation | 0.033333 | 152b2a1b12da65fb |
| sudoku | finite_geometric | 101 | 16 | 0.0 | 0.001 | 1000 | completed | validation | 0.000000 | a7029e2710e6c02b |
| sudoku | finite_geometric | 101 | 4 | 0.0 | 0.001 | 1000 | completed | validation | 0.044444 | 18558b27954dafbd |
| sudoku | finite_geometric | 101 | 8 | 0.0 | 0.001 | 1000 | completed | validation | 0.061111 | b0861b894cce818a |

## smoke

| Задача | Метод | Seed | H | β | LR | Updates | Status | Split | Solved % | Config hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maze | binomial | 101 | 4 | 0.0 | 0.001 | 6 | completed | validation | 0.000000 | 526e349061e469ae |
| sudoku | binomial | 101 | 4 | 0.0 | 0.001 | 6 | completed | validation | 0.000000 | 31e9e04834e2bece |

## Время и память

| Задача | Метод | Train s, mean | Elapsed s, mean | Forward μs/example, mean | Peak train MiB, max |
| --- | --- | --- | --- | --- | --- |
| maze | onehop | 282.758 | 300.450 | 38.16 | 225.8 |
| maze | binomial | 593.497 | 622.771 | 74.86 | 309.4 |
| maze | exponential_truncated | 593.983 | 623.574 | 75.38 | 309.4 |
| maze | finite_geometric | 592.567 | 622.375 | 74.90 | 309.4 |
| maze | learned_polynomial | 651.333 | 681.405 | 75.10 | 317.4 |
| maze | recurrent | 1520.721 | 1565.486 | 253.97 | 1199.3 |
| sudoku | onehop | 288.711 | 552.981 | 36.87 | 239.6 |
| sudoku | binomial | 351.815 | 633.544 | 52.05 | 270.0 |
| sudoku | exponential_truncated | 352.892 | 633.641 | 52.09 | 270.0 |
| sudoku | finite_geometric | 352.860 | 635.397 | 52.15 | 270.0 |
| sudoku | learned_polynomial | 377.584 | 663.793 | 51.83 | 280.2 |
| sudoku | recurrent | 1543.858 | 2446.470 | 253.03 | 1578.4 |

## Пороги качества

| Задача | Метод | Порог % | Достигли | Первый passing step по пяти seeds |
| --- | --- | --- | --- | --- |
| maze | onehop | 90 | 0/5 | ценз., ценз., ценз., ценз., ценз. |
| maze | binomial | 90 | 5/5 | 4250, 5000, 4500, 5000, 4000 |
| maze | exponential_truncated | 90 | 5/5 | 3250, 3500, 4000, 3750, 3000 |
| maze | finite_geometric | 90 | 5/5 | 3250, 2250, 2750, 2250, 2250 |
| maze | learned_polynomial | 90 | 5/5 | 2750, 2500, 2250, 2750, 2750 |
| maze | recurrent | 90 | 4/5 | 25000, 22500, ценз., 21250, 16250 |
| sudoku | onehop | 50 | 0/5 | ценз., ценз., ценз., ценз., ценз. |
| sudoku | onehop | 90 | 0/5 | ценз., ценз., ценз., ценз., ценз. |
| sudoku | binomial | 50 | 0/5 | ценз., ценз., ценз., ценз., ценз. |
| sudoku | binomial | 90 | 0/5 | ценз., ценз., ценз., ценз., ценз. |
| sudoku | exponential_truncated | 50 | 0/5 | ценз., ценз., ценз., ценз., ценз. |
| sudoku | exponential_truncated | 90 | 0/5 | ценз., ценз., ценз., ценз., ценз. |
| sudoku | finite_geometric | 50 | 0/5 | ценз., ценз., ценз., ценз., ценз. |
| sudoku | finite_geometric | 90 | 0/5 | ценз., ценз., ценз., ценз., ценз. |
| sudoku | learned_polynomial | 50 | 0/5 | ценз., ценз., ценз., ценз., ценз. |
| sudoku | learned_polynomial | 90 | 0/5 | ценз., ценз., ценз., ценз., ценз. |
| sudoku | recurrent | 50 | 5/5 | 4500, 4000, 3250, 3250, 3000 |
| sudoku | recurrent | 90 | 5/5 | 21250, 21750, 20250, 21250, 21250 |

## Pointer: все 36 оценок

| Условие | Метод | Seed | H | Conditional accuracy | Terminal mass | Unreached accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| iid_depth2 | binomial | 11 | 6 | 1.0 | 0.9047577455639839 | 1.0 |
| iid_depth3 | binomial | 11 | 6 | 1.0 | 0.666685875505209 | 1.0 |
| depth6_fixed_budget | binomial | 11 | 6 | 0.997802734375 | 0.016174939344637096 | 0.0 |
| depth6_scaled_budget | binomial | 11 | 12 | 1.0 | 0.6131252609193325 | 1.0 |
| iid_depth2 | finite_geometric | 11 | 6 | 1.0 | 0.7616430856287479 | 1.0 |
| iid_depth3 | finite_geometric | 11 | 6 | 1.0 | 0.5580370426177979 | 1.0 |
| depth6_fixed_budget | finite_geometric | 11 | 6 | 0.99951171875 | 0.10870588477700949 | 0.0 |
| depth6_scaled_budget | finite_geometric | 11 | 12 | 1.0 | 0.521666020154953 | 1.0 |
| iid_depth2 | exponential_truncated | 11 | 6 | 1.0 | 0.844941969960928 | 1.0 |
| iid_depth3 | exponential_truncated | 11 | 6 | 1.0 | 0.6066183745861053 | 1.0 |
| depth6_fixed_budget | exponential_truncated | 11 | 6 | 0.999267578125 | 0.059429556829854846 | 0.0 |
| depth6_scaled_budget | exponential_truncated | 11 | 12 | 1.0 | 0.5613033138215542 | 1.0 |
| iid_depth2 | binomial | 23 | 6 | 1.0 | 0.9047612957656384 | 1.0 |
| iid_depth3 | binomial | 23 | 6 | 1.0 | 0.6666817776858807 | 1.0 |
| depth6_fixed_budget | binomial | 23 | 6 | 0.9990234375 | 0.0160757644334808 | 0.0 |
| depth6_scaled_budget | binomial | 23 | 12 | 1.0 | 0.6130776330828667 | 1.0 |
| iid_depth2 | finite_geometric | 23 | 6 | 1.0 | 0.7616477347910404 | 1.0 |
| iid_depth3 | finite_geometric | 23 | 6 | 1.0 | 0.558037307113409 | 1.0 |
| depth6_fixed_budget | finite_geometric | 23 | 6 | 0.999755859375 | 0.10861821891739964 | 0.0 |
| depth6_scaled_budget | finite_geometric | 23 | 12 | 1.0 | 0.5216166712343693 | 0.99951171875 |
| iid_depth2 | exponential_truncated | 23 | 6 | 1.0 | 0.8449468016624451 | 1.0 |
| iid_depth3 | exponential_truncated | 23 | 6 | 1.0 | 0.6066151298582554 | 1.0 |
| depth6_fixed_budget | exponential_truncated | 23 | 6 | 0.99951171875 | 0.059330603340640664 | 0.0 |
| depth6_scaled_budget | exponential_truncated | 23 | 12 | 1.0 | 0.5612489506602287 | 1.0 |
| iid_depth2 | binomial | 37 | 6 | 1.0 | 0.9047605022788048 | 1.0 |
| iid_depth3 | binomial | 37 | 6 | 1.0 | 0.6666844934225082 | 1.0 |
| depth6_fixed_budget | binomial | 37 | 6 | 0.99951171875 | 0.016007461352273822 | 0.0 |
| depth6_scaled_budget | binomial | 37 | 12 | 1.0 | 0.6130379065871239 | 1.0 |
| iid_depth2 | finite_geometric | 37 | 6 | 1.0 | 0.761647529900074 | 1.0 |
| iid_depth3 | finite_geometric | 37 | 6 | 1.0 | 0.5580350160598755 | 1.0 |
| depth6_fixed_budget | finite_geometric | 37 | 6 | 1.0 | 0.10856986744329333 | 0.0 |
| depth6_scaled_budget | finite_geometric | 37 | 12 | 1.0 | 0.5215875655412674 | 0.999267578125 |
| iid_depth2 | exponential_truncated | 37 | 6 | 1.0 | 0.8449463546276093 | 1.0 |
| iid_depth3 | exponential_truncated | 37 | 6 | 1.0 | 0.6066162064671516 | 1.0 |
| depth6_fixed_budget | exponential_truncated | 37 | 6 | 1.0 | 0.05927104502916336 | 0.0 |
| depth6_scaled_budget | exponential_truncated | 37 | 12 | 1.0 | 0.561207965016365 | 0.999755859375 |

## Идентификаторы данных

Manifest содержит исходные CSV hashes и hashes полученных NPZ. Байты NPZ могут зависеть от версии упаковки; воспроизведение на другом окружении следует проверять также по содержимому массивов.

```json
{
  "maze": {
    "data_hash": "cc9d4a77297250f178f77c206e0b7c24254a62689d33b2e53e937d88b9c6a0cb",
    "files": {
      "test.npz": "e6e379a3876c0fe551d21fc2e956c277b61fb8f8a41e6b9824e16ab31479f636",
      "train.npz": "ab6b4001ea65db465aa99100f916d30af4bece6738b9ddea47531e5c41a1fc51",
      "validation.npz": "0d9e862915de89f54d35fafeaec1b726b22bcfeb9ae6d8d66377656246e5b8b0"
    },
    "labels_available_to_model": false,
    "long_path_boundary": 32.0,
    "maze_split": "D4 canonical topology",
    "no_augmentation": true,
    "pos_weight": 2.624768496230807,
    "source": "independent generator v1",
    "spec": {
      "clue_fraction": 0.65,
      "data_seed": 20261009,
      "data_source": "generated",
      "data_tag": "maze-8x8-pilot-v1",
      "generator": "dfs",
      "size": 8,
      "task": "maze",
      "test_count": 1000,
      "train_count": 10000,
      "val_count": 1000
    },
    "splits": {
      "test": 1000,
      "train": 10000,
      "validation": 1000
    },
    "statistics": {
      "test": {
        "path_length_max": 50,
        "path_length_mean": 16.628,
        "path_length_quantiles": [
          8.0,
          15.0,
          24.0,
          32.0,
          37.0
        ]
      },
      "train": {
        "path_length_max": 55,
        "path_length_mean": 16.6563,
        "path_length_quantiles": [
          8.0,
          15.0,
          24.0,
          32.0,
          36.0
        ]
      },
      "validation": {
        "path_length_max": 52,
        "path_length_mean": 16.881,
        "path_length_quantiles": [
          8.0,
          15.0,
          25.0,
          32.10000000000002,
          36.0
        ]
      }
    },
    "sudoku_grouping": null,
    "unique_sudoku_solutions": false
  },
  "sudoku": {
    "cross_split_exact_duplicate_rows": 0,
    "data_hash": "9b17eb5fd873bc3cd719c5e7f899f9e503771c3c9d96a354f03bdd3710a3603a",
    "download_url": "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1",
    "files": {
      "test.npz": "2099b8a652c0d84a893c39bccc0513bc76b1eba027a37dd1ce5185d019f42f67",
      "train.npz": "6015664485b7edd826ce0a96ed8583766fc58c1c92e3a3981856fa34af1a1ec6",
      "validation.npz": "c7a1ad4347e660c9a4664dfce943938754ec09ccec1da740ec0eef3776e8b27f"
    },
    "full_symmetry_audit": false,
    "labels_available_to_model": false,
    "license": "No explicit dataset license found in the inspected source; external private research input, not redistributed",
    "no_augmentation": true,
    "official_splits_preserved": true,
    "source": "Recurrent Relational Networks official sudoku-hard data",
    "source_url": "https://github.com/rasmusbergpalm/recurrent-relational-networks/blob/master/tasks/sudoku/data.py",
    "spec": {
      "clue_fraction": 0.65,
      "data_seed": 20261009,
      "data_source": "rrn",
      "data_tag": "rrn-official-v1",
      "generator": "dfs",
      "size": 9,
      "task": "sudoku",
      "test_count": 5000,
      "train_count": 100000,
      "val_count": 5000
    },
    "splits": {
      "test": 18000,
      "train": 180000,
      "validation": 18000
    },
    "statistics": {
      "test": {
        "clue_count_mean": 25.5,
        "source_sha256": "1b14d68f58de6f64b3db4032eaf4ea68783471aceb8ff10306b92d0479c6bc99"
      },
      "train": {
        "clue_count_mean": 25.5,
        "source_sha256": "797f8c6dddf9d138d67b86ad22990a839b951b4358c13d3aaf0d91171d30d8b5"
      },
      "validation": {
        "clue_count_mean": 25.5,
        "source_sha256": "950b2f167a6e71d24a368cd1d1e2acefc1d54faf19507ed47dc24df3e45f2eef"
      }
    },
    "unique_solution": "as supplied; sampled solver audit saved separately"
  }
}
```

## Завершённая финальная MLP-серия

20 дополнительных финальных обучений опубликованы отдельно от архивных152 строк: [подробный отчёт](../REPORT_MLP_FINAL_RU.md), [40 строк on/off по seeds](../results/tables/mlp_final_per_seed.csv), [парные интервалы](../results/tables/mlp_final_contrasts.csv).

| Задача | Фильтр | Без MLP, % | С MLP, % | Δ, п.п. | 95% CI Δ, п.п. |
| --- | --- | --- | --- | --- | --- |
| maze | Binomial | 98.7600 ± 0.1817 | 99.7200 ± 0.1789 | +0.9600 | [+0.7400; +1.1800] |
| maze | Exponential | 99.5800 ± 0.2168 | 99.9400 ± 0.0894 | +0.3600 | [+0.1800; +0.5600] |
| sudoku | Binomial | 5.2278 ± 1.0799 | 38.9789 ± 1.5801 | +33.7511 | [+32.1644; +34.9556] |
| sudoku | Exponential | 5.7622 ± 0.9416 | 39.6578 ± 1.3395 | +33.8956 | [+33.3889; +34.4022] |
