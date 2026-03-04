from ultralytics import YOLO

model = YOLO('/Home/cr/Mamba-YOLO/output_last/MB-YOLO-T+SE_IN+DSM_ms0.3_bs4/train_GPU3/weights/best.pt')

results = model.val(
    data='/Home/cr/Mamba-YOLO/ultralytics/cfg/datasets/S-UODAC-t.yaml',
    project='/Home/cr/Mamba-YOLO/output_last/MB-YOLO-T+SE_IN+DSM_ms0.3_bs4',
    name='test_GPU3'
)
