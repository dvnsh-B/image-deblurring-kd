import torch.nn.functional as F

def pad_to_multiple(img, multiple=16):
    h, w = img.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    # Padding format: (left, right, top, bottom)
    padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return padded_img, pad_h, pad_w

def crop_to_original(img, pad_h, pad_w):
    if pad_h > 0:
        img = img[..., :-pad_h, :]
    if pad_w > 0:
        img = img[..., :, :-pad_w]
    return img
