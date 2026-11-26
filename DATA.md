# Data Preparation

DEFDVO is trained and evaluated on the equirectangular (ERP) images of TartanAirV2 (`lcam_equirect`) and evaluated
on the real-world 360DVO dataset.

## TartanAirV2

Training and `evaluate_tartan.py` use the left equirectangular camera (`lcam_equirect`) with the poses in
`pose_lcam_front.txt`. Download them with the TartanAir toolkit ([tartanair.org](https://tartanair.org)), installed
in a separate environment with `pip install tartanair`:

```python
import tartanair as ta

ta.init('./tartanair2')
envs = ta.get_all_data()['env']
ta.download(env=envs, difficulty=['easy', 'hard'], modality=['image', 'depth'],
            camera_name=['lcam_equirect'], unzip=True, delete_zip=True)
# provides pose_lcam_front.txt
ta.download(env=envs, difficulty=['easy', 'hard'], modality=['image'],
            camera_name=['lcam_front'], unzip=True, delete_zip=True)
```

This downloads every trajectory of every environment (several TB); training and evaluation use only easy P000 and
hard P000-P001.

Expected structure:

```
tartanair2/
└── <Scene>/
    ├── Data_easy/P000/
    │   ├── image_lcam_equirect/000000_lcam_equirect_image.png ...
    │   ├── depth_lcam_equirect/000000_lcam_equirect_depth.png ...
    │   └── pose_lcam_front.txt
    ├── Data_hard/P000/   (same layout)
    └── Data_hard/P001/   (same layout)
```

Build the training index (not needed for evaluation):

```bash
python create_tartanairv2_pickle.py --root ./tartanair2
```

This writes `tartanairv2_lcam_equirect_subset.pkl`. Training uses easy P000 and hard P000-P001 of every scene where
available; the 19 evaluation trajectories of Set 1 and Set 2 are always excluded.

## 360DVO

Download the dataset released by the 360DVO authors on
[Hugging Face](https://huggingface.co/datasets/chris1004336379/360DVO) (CC BY-NC-SA 4.0):

```bash
pip install -U huggingface_hub
hf download chris1004336379/360DVO --repo-type dataset --local-dir 360DVO/Data
```

`evaluate_360dvo.py` reads the dataset from `./360DVO/Data` (change with `--datapath`) and expects:

```
360DVO/Data/
├── Sequences/
│   └── <sequence>/0001.jpg ...
└── GroundTruth/
    └── <sequence>.txt
```

`<sequence>` is one of the 20 sequence names in `evaluate_360dvo.py` (e.g. `bridge_night`). Each ground-truth file
has one `tx ty tz qx qy qz qw` line per image.
