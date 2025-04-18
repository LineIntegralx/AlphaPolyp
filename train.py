import os, csv, gc, datetime
import numpy as np
import albumentations as albu
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW

from model_architecture.DiceLoss import dice_metric_loss
from model_architecture.Model   import create_model
from model_architecture.ImageLoader2D  import load_images_masks_from_drive

# ── 1) USER CONFIG —————————————————————————————————————————————————————————
drive_base     = '/content/drive/MyDrive/extracted_folder/synth-colon'
real_img_dir  = os.path.join(drive_base, 'cyclegan_images')
real_mask_dir = os.path.join(drive_base, 'masks')
synth_img_dir  = os.path.join(drive_base, 'images')
synth_mask_dir = os.path.join(drive_base, 'masks')
csv_labels     = 'cleaned_labels.csv'        # CSV with columns: Filename,Volume,x,y,z

img_size       = 500
filters        = 17
batch_size     = 8
seed           = 58800

# how many epochs for each phase (tune as needed)
epochs_phase1  = 15  # head-only
epochs_phase2  = 30  # fine-tune all

# path to your pretrained RAPUNet checkpoint (segmentation only)
pretrained_ckpt = 'rapunet_pretrained.h5'

log_root       = './logs'
os.makedirs(log_root, exist_ok=True)


# ── 2) READ REGRESSION LABEL CSV ———————————————————————————————————————————————
label_map = {}
with open(csv_labels, newline='') as f:
    for row in csv.DictReader(f):
        label_map[row['Filename']] = [
            float(row['Volume']),
            float(row['x']),
            float(row['y']),
            float(row['z'])
        ]

def get_reg_labels(file_list):
    """Return an [N,4] array of [vol,x,y,z] for each filename."""
    regs = []
    for fname in file_list:
        base    = os.path.splitext(fname)[0]        
        csv_key = f"{base}_labeled.obj"             
        regs.append(label_map.get(csv_key, [0.0,0.0,0.0,0.0]))
    return np.array(regs, dtype=np.float32)


# ── 3) LOAD REAL & SYNTHETIC IMAGES + MASKS ——————————————————————————————
real_files  = sorted(os.listdir(real_img_dir))
synth_files = sorted(os.listdir(synth_img_dir))

X_real,  Y_real_mask  = load_images_masks_from_drive(real_img_dir,  real_mask_dir,  img_size)
X_synth, Y_synth_mask = load_images_masks_from_drive(synth_img_dir, synth_mask_dir, img_size)

Y_real_reg  = get_reg_labels(real_files)
Y_synth_reg = get_reg_labels(synth_files)


# ── 4) INTERLEAVE REAL ↔ SYNTHETIC ——————————————————————————————————————————
def interleave(X1, M1, R1, X2, M2, R2):
    Xc, Mc, Rc = [], [], []
    n = min(len(X1), len(X2))
    for i in range(n):
        Xc.append(X1[i]); Mc.append(M1[i]); Rc.append(R1[i])
        Xc.append(X2[i]); Mc.append(M2[i]); Rc.append(R2[i])
    # append any leftovers
    if len(X1) > n:
        Xc.extend(X1[n:]); Mc.extend(M1[n:]); Rc.extend(R1[n:])
    if len(X2) > n:
        Xc.extend(X2[n:]); Mc.extend(M2[n:]); Rc.extend(R2[n:])
    return np.array(Xc), np.array(Mc), np.array(Rc)

X_all, Y_all_mask, Y_all_reg = interleave(
    X_real,  Y_real_mask,  Y_real_reg,
    X_synth, Y_synth_mask, Y_synth_reg
)

# ── 5) SPLIT TRAIN / VALID ——————————————————————————————————————————————
x_train, x_val, y_train_mask, y_val_mask, y_train_reg, y_val_reg = \
    train_test_split(
        X_all, Y_all_mask, Y_all_reg,
        test_size=0.1, shuffle=True, random_state=seed
    )

# ── 6) AUGMENTATION ——————————————————————————————————————————————————————
aug = albu.Compose([
    albu.HorizontalFlip(),
    albu.VerticalFlip(),
    albu.ColorJitter(brightness=(0.6,1.6),contrast=0.2,saturation=0.1,hue=0.01,always_apply=True),
    albu.Affine(scale=(0.5,1.5),translate_percent=(-0.125,0.125),rotate=(-180,180),shear=(-22.5,22),always_apply=True),
])

def augment_batch(X, M, R):
    """Apply albumentations to images & masks; leave regression labels unchanged."""
    Xa, Ma, Ra = [], [], []
    for img, msk, reg in zip(X, M, R):
        a = aug(image=(img*255).astype(np.uint8),
                mask =(msk*255).astype(np.uint8))
        Xa.append(a['image'] / 255.0)
        Ma.append(a['mask']  / 255.0)
        Ra.append(reg)
    return ( np.array(Xa, dtype=np.float32),
             np.expand_dims(np.array(Ma, dtype=np.float32), -1),
             np.array(Ra, dtype=np.float32) )


# ── 7) BUILD MODEL & LOAD PRETRAINED RAPUNet WEIGHTS —————————————————————————
model = create_model(img_size, img_size, 3, 1, filters)

if os.path.exists(pretrained_ckpt):
    model.load_weights(pretrained_ckpt, by_name=True, skip_mismatch=True)
    print('Loaded pretrained RAPUNet weights.')

# ── CALLBACKS ─────────────────────────────────────────────────────────────────
run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
callbacks = [
    CSVLogger    (f'{log_root}/train_{run_id}.csv'),
    TensorBoard  (log_dir=f'{log_root}/tb_{run_id}'),
    ModelCheckpoint('alphapolyp_optimized_model.h5',
                    monitor='val_segmentation_output_loss',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_segmentation_output_loss',
                      factor=0.5, patience=8, verbose=1)
]


# ── 8) PHASE 1: TRAIN REGRESSION HEAD ONLY ——————————————————————————————————
# freeze everything except regression head
for layer in model.layers:
    layer.trainable = 'regression_output' in layer.name

model.compile(
    optimizer=AdamW(1e-4, weight_decay=1e-6),
    loss={'segmentation_output': dice_metric_loss,
          'regression_output'  : 'mse'},
    loss_weights={'segmentation_output':1.0,
                  'regression_output'  :1.0}
)

print('Phase1: training regression head only')
for epoch in range(epochs_phase1):
    Xa, Ma, Ra = augment_batch(x_train, y_train_mask, y_train_reg)
    model.fit(Xa,
              {'segmentation_output': Ma,
               'regression_output'  : Ra},
              validation_data=(x_val,
                               {'segmentation_output': y_val_mask,
                                'regression_output'  : y_val_reg}),
              epochs=1, batch_size=batch_size,
              callbacks=callbacks, verbose=1)
    gc.collect()


# ── 9) PHASE 2: FINE‑TUNE ENTIRE NETWORK ———————————————————————————————————
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=AdamW(1e-5, weight_decay=1e-6),
    loss={'segmentation_output': dice_metric_loss,
          'regression_output'  : 'mse'},
    loss_weights={'segmentation_output':1.0,
                  'regression_output'  :1.0}
)

print('Phase2: fine-tuning all layers')
for epoch in range(epochs_phase2):
    Xa, Ma, Ra = augment_batch(x_train, y_train_mask, y_train_reg)
    model.fit(Xa,
              {'segmentation_output': Ma,
               'regression_output'  : Ra},
              validation_data=(x_val,
                               {'segmentation_output': y_val_mask,
                                'regression_output'  : y_val_reg}),
              epochs=1, batch_size=batch_size,
              callbacks=callbacks, verbose=1)
    gc.collect()

print('Training complete — alphapolyp_optimized_model.h5 saved.')
