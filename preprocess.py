import torchvision.transforms as T
from pathlib import Path
import PIL

## [collect imgs]
dir = Path.cwd()

img_path_neg = dir/"INRIAPerson/train_64x128_H96/neg.lst"
with open(img_path_neg, 'r') as f:
    n_paths = [Path("INRIAPerson")/i[:-1].capitalize() for i in f]

img_path_pos = dir/"INRIAPerson/train_64x128_H96/pos.lst"
with open(img_path_pos, 'r') as f:
    p_paths = [Path("INRIAPerson/96X160H96/Train/pos")/Path(i[:-1]).name for i in f]

for i in range(len(n_paths)):
  if not n_paths[i].exists():
    n_paths[i] = n_paths[i].parent/str(n_paths[i].name).capitalize()
    assert n_paths[i].exists(), f"{n_paths[i]} " + 'not exist'

for i in range(len(p_paths)):
  if not p_paths[i].exists():   
    p_paths[i] = p_paths[i].parent/str(p_paths[i].name).capitalize()
    assert p_paths[i].exists(), f"{p_paths[i]} " + 'not exist'

## [collect imgs]

## [crop neg imgs]
print('crop neg imgs')
cropper_neg = T.RandomCrop(size=(128, 64))
crop_path_neg = Path("INRIAPerson/train_64x128_H96/neg_crop")
crop_path_neg.mkdir(exist_ok=True)
names_all_neg = []
for i,path in enumerate(n_paths):
    orig_img = PIL.Image.open(path)
    names = [crop_path_neg/Path(str(Path(path).stem)+f"_crop_{j}.png") for j in range(10)]
    names_all_neg.extend([str(i) for i in names])
    crops = [cropper_neg(orig_img) for _ in range(10)]
    for name,crop in zip(names, crops):
        crop.save(name)
lst_path = crop_path_neg.parent/Path("neg_crop.lst")
lst_path.touch(exist_ok=True)
lst_path.write_text("\n".join(names_all_neg)+"\n")
## [crop neg imgs]

## [crop pos imgs]
print('crop pos imgs')
cropper_pos = T.CenterCrop(size=(128,64))
crop_path_pos = Path("INRIAPerson/train_64x128_H96/pos_crop")
crop_path_pos.mkdir(exist_ok=True)
names_all_pos = []
for i,path in enumerate(p_paths):
    orig_img = PIL.Image.open(path)
    name = crop_path_pos/Path(str(Path(path).stem)+"_crop.png")
    names_all_pos.append(str(name))
    crop = cropper_pos(orig_img)
    crop.save(name)
lst_path = crop_path_pos.parent/Path("pos_crop.lst")
lst_path.touch(exist_ok=True)
lst_path.write_text("\n".join(names_all_pos)+"\n")
## [crop pos imgs]

print('cropped saved')