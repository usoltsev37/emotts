FOR EXPERIMENTS:

---------------------------------------
I) FASTSPEECH2:

1) python train_fastspeech2_voiceprint.py --config configs/fastspeech_base/fastspeech2_gst.yml

2а) в последнем чекпойнте убрать номер итерации из начала названия файла:
к примеру, было 
checkpoints/fastspeech2/fastspeech2/500000_fastspeech2_model.pth
стало
checkpoints/fastspeech2/fastspeech2/fastspeech2_model.pth

2б) python train_fastspeech2_voiceprint.py --config configs/fastspeech_base/fastspeech2_gst_tune.yml

#checkpoint path: checkpoints/fastspeech2/fastspeech2/

---------------------------------------
II) FASTSPEECH2 WITHOUT PITCH/ENERGY:

1) python train_fastspeech2_duration.py --config configs/fastspeech_reduced_va/fastspeech2_gst_reduced_va.yml

2а) в последнем чекпойнте убрать номер итерации из начала названия файла:
к примеру, было 
checkpoints/fastspeech2_reduced_va/fastspeech2/500000_fastspeech2_model.pth
стало
checkpoints/fastspeech2_reduced_va/fastspeech2/fastspeech2_model.pth

2б) python train_fastspeech2_duration.py --config configs/fastspeech_reduced_va/fastspeech2_gst_reduced_va_tune.yml

#checkpoint path: checkpoints/fastspeech2_reduced_va/fastspeech2/

---------------------------------------
III) TACOTRON INFLATED:

1) python train_non_attentive_voiceprint.py --config configs/nat_inflated/nat_inflated.yml

2а) в последнем чекпойнте убрать номер итерации из начала названия файла:
к примеру, было 
checkpoints/nat_inflated/feature/500000_feature_model.pth
стало
checkpoints/nat_inflated/feature/feature_model.pth

2б) python train_non_attentive_voiceprint.py --config configs/nat_inflated/nat_inflated_tune.yml

#checkpoint path: checkpoints/nat_inflated/feature/

---------------------------------------
IV) TACOTRON VA BEFORE DURATION:

1) python train_non_attentive_voiceprint_variance.py --config configs/nat_va_before_duration/nat_va_before_duration.yml

2а) в последнем чекпойнте убрать номер итерации из начала названия файла:
к примеру, было 
checkpoints/nat_va_before_duration/feature/500000_feature_model.pth
стало
checkpoints/nat_inflated/feature/feature_model.pth

2б) python train_non_attentive_voiceprint_variance.py --config configs/nat_va_before_duration/nat_va_before_duration_tune.yml

#checkpoint path: checkpoints/nat_va_before_duration/feature/

---------------------------------------
V) TACOTRON VA AFTER DURATION (BEROFE DECODER):

python train_non_attentive_voiceprint_variance_after_duration.py --config configs/nat_va_after_duration/nat_va_after_duration.yml

2а) в последнем чекпойнте убрать номер итерации из начала названия файла:
к примеру, было 
checkpoints/nat_va_after_duration/feature/500000_feature_model.pth
стало
checkpoints/nat_va_after_duration/feature/feature_model.pth

2б) python train_non_attentive_voiceprint_variance_after_duration.py --config configs/nat_va_after_duration/nat_va_after_duration_tune.yml

#checkpoint path: checkpoints/nat_va_after_duration/feature/1_feature_model.pth
