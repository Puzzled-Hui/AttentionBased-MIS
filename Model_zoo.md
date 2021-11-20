# Model Zoo
This Chapter provides some pre-trained models and important hyperparameters experiments results.

The description for Attention Module is clarified in README.md

## Attention Based Models for Medical Image Segmentation

- For BraTS 2018 dataset

| Backbone | AttentionModule | Dice | Pretrained Model |
| ------------- | ------------- | ------------- | ------------- |
| 3D UNet | None | 0.7685 | [Link](https://drive.google.com/open?id=1osY3tvucDRS5Vb2-gTLrAmQn99hDcNy5) |
| 3D UNet | AnatomyNet | 0.7768 | [Link](https://drive.google.com/open?id=14ZhWTSFO8Lf-pPraou_GW0JW30AOHwfI) |
| 3D UNet | PENet | 0.7818 | [Link](https://drive.google.com/open?id=1oOAAuPKmV3i3tXkh5ZnVTQTm3QMYeRf9) |
| 3D UNet | AG | 0.7748 | [Link](https://drive.google.com/open?id=1XhS6TSl4EiopE88iKgEB2oeS8TV4E9zB)|
| 3D UNet | csA Net | 0.7831 | [Link](https://drive.google.com/open?id=1Ct4oYipJnN9FxsmfJQUEvom9rQoLjheq) |
| 3D UNet | scA Net | 0.7842 | [Link](https://drive.google.com/open?id=1Ct4oYipJnN9FxsmfJQUEvom9rQoLjheq)|
| 3D UNet | PANet | **0.7970** | [Link](https://drive.google.com/open?id=1-Y5XJboLERag2qhb9TkDnwG-nMklpf_i) |

| Backbone | AttentionModule | Dice | Pretrained Model |
| ------------- | ------------- | ------------- | ------------- |
| VNet | None | 0.7680 | [Link](https://drive.google.com/open?id=1eLrsISBgP6FwDs6vOI1e83SclIwDqRz6) |
| VNet | AnatomyNet | 0.7777 | [Link](https://drive.google.com/open?id=1Q1x33llv2-CrXGDLtn7C9mSujysNZKUy) |
| VNet | csA Net | 0.7804 | [Link](https://drive.google.com/open?id=1FOxJwN-lZB9rEoKWJt1clfsF7ApEbCJF) |
| VNet | scA Net | 0.7915 | [Link](https://drive.google.com/open?id=1cStZuduKd-rSo3yn3HdHnd2bbB2j2l1P)|
| VNet | PANet | **0.8034** | [Link](https://drive.google.com/open?id=1BZS_hbYz_JCHhNqJ5IziAKfsZmrv0QSt) |


- For MALC dataset

| Backbone | AttentionModule | Dice | Pretrained Model |
| ------------- | ------------- | ------------- | ------------- |
| 3D UNet | None | 0.8372 | [Link](https://drive.google.com/open?id=1fxf4i2dmMCMfPC3EZ91x7KlTJF3NvRJW) |
| 3D UNet | AnatomyNet | 0.8420 | [Link](https://drive.google.com/open?id=1XdaXimWYA0pmKO2nT-CBl9A50hfmrtoi) |
| 3D UNet | PENet | 0.8566 | [Link](https://drive.google.com/open?id=1Vonujfmjz8X1FAui9POf0sHXoI9Uhhtd) |
| 3D UNet | AG | 0.8449 | [Link](https://drive.google.com/open?id=1P8MeqfPUaU7aTFIzRnqARoL1_pNlhes_) |
| 3D UNet | csA Net | 0.8548 | [Link](https://drive.google.com/open?id=1MEVgTAEYVQIbOcm21vK1QAA-qg9hJohA) |
| 3D UNet | scA Net | 0.8698 | [Link](https://drive.google.com/open?id=1RvETQFSzkDlxcfkshfYQt9qgtBzsUVup) |
| 3D UNet | PANet | **0.8767** | [Link](https://drive.google.com/open?id=1YqVs20jJuoce_hDHUf-HxfrIC3vuzAcd) |

| Backbone | AttentionModule | Dice | Pretrained Model |
| ------------- | ------------- | ------------- | ------------- |
| VNet | None | 0.8336 | [Link](https://drive.google.com/open?id=1wj665V_j2MbZdtNhbvlU8HBYYsRIGb3j) |
| VNet | AnatomyNet | 0.8326 | [Link](https://drive.google.com/open?id=115YvXW5UDzF5Sl7QUvwgLJGCsdrBSkS9) |
| VNet | csA Net | 0.8611 | [Link](https://drive.google.com/open?id=1uYgBtS-P6kQSkQroJrlFMNBJM6IsxpFp) |
| VNet | scA Net | 0.8757 | [Link](https://drive.google.com/open?id=1w7ogvyShPKlFF4O96OatF2aOL5zY35lD) |
| VNet | PANet | **0.8898** | [Link](https://drive.google.com/open?id=19UTgxiyb317_0IpWExwjx6hJRANwdH2q) |


- For HVSMR dataset

| Backbone | AttentionModule | Dice | Pretrained Model |
| ------------- | ------------- | ------------- | ------------- |
| 3D UNet | None | 0.8062 | [Link](https://drive.google.com/open?id=14OFiMZhzwGmTt50e0CZ08xgpmF8wAQ3U) |
| 3D UNet | AnatomyNet | 0.8165 | [Link](https://drive.google.com/open?id=1rd_i3xHih3assL_OneSU4KK7N3E8MG5e) |
| 3D UNet | PENet | 0.7078 | [Link](https://drive.google.com/open?id=1LWWNSVsNrq4awVocoo-GY5Qfm-KI8SJb) |
| 3D UNet | AG | 0.8139 | [Link](https://drive.google.com/open?id=1BxYZ52cuWllmsHxh_k9UFWW3TFejJbnd) |
| 3D UNet | csA Net | 0.8077 | [Link](https://drive.google.com/open?id=1rqfIOo5Ug27MZVxtUrSYAYewAWImu5Vw) |
| 3D UNet | scA Net | 0.8313 | [Link](https://drive.google.com/open?id=1IVjBHDDWjdlpCGbulK8SAeW2Ik_Kwtur) |
| 3D UNet | PANet | **0.8443** | [Link](https://drive.google.com/open?id=19ebVZX-CRF8JshDDT6qnZDcx6V9UEnuV) |

| Backbone | AttentionModule | Dice | Pretrained Model |
| ------------- | ------------- | ------------- | ------------- |
| VNet | None | 0.8003 | [Link](https://drive.google.com/open?id=1KjH_GlsIiXDHkDr4CH92hy--9kTCqGSw) |
| VNet | AnatomyNet | 0.8082 | [Link](https://drive.google.com/open?id=1jacGriVv2Y4CI3m6CwJ4hTuLa-nyWEku) |
| VNet | csA Net | 0.8233 | [Link](https://drive.google.com/open?id=1zXVVwW8wjBjNKgdubPjjEMQ-0bRsyZ61) |
| VNet | scA Net | 0.8046 | [Link](https://drive.google.com/open?id=1Yp6QDD26nXWAu25CZGfGTdvrvQvcUlit) |
| VNet | PANet | **0.8361** | [Link](https://drive.google.com/open?id=1eG6AM91i40lWswzab2tG6CBoeoYQM_A-) |


