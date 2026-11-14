from dataclasses import asdict, dataclass
import numpy as np
from scipy.ndimage import zoom

VIEW_NAMES = ('main128', 'lower_context', 'upper_context', 'texture32')

@dataclass(frozen=True)
class SamplingConfig:
    slab_axis: int = 2
    thickness: int = 26
    overlap: int = 13
    extra_width: int = 12 
    plane_size: int = 128
    texture_size: int = 32
    texture_jitter: int = 4

    def __post_init__(self):
        if self.slab_axis not in (0, 1, 2):
            raise ValueError('slab_axis must be 0, 1 or 2; confirm volume orientation first')
        if not 15 <= self.thickness <= 26 or not 7 <= self.extra_width <= 20:
            raise ValueError('thickness must be 15..26 and extra_width 7..20 voxels')
        if not 0 <= self.overlap < self.thickness:
            raise ValueError('overlap must be nonnegative and smaller than thickness')
        if self.plane_size < 16 or self.texture_size != 32 or self.texture_jitter < 0:
            raise ValueError('Invalid plane_size, texture_size or jitter')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        # Earlier checkpoints used completely external slabs. Never silently change them.
        values = dict(values)
        values.setdefault('overlap', 0)
        return cls(**values)

def crop_padded(volume, mask, start, size):
    start = np.asarray(start, dtype=int)
    size = np.asarray(size, dtype=int)
    shape = np.asarray(mask.shape)
    lo, hi = np.maximum(start, 0), np.minimum(start + size, shape)
    image = np.zeros(tuple(size) + (volume.shape[-1],), dtype=np.float32)
    seg = np.zeros(tuple(size), dtype=np.uint8)
    if np.all(hi > lo):
        src = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        dst = tuple(slice(int(a), int(b)) for a, b in zip(lo - start, hi - start))
        image[dst], seg[dst] = volume[src], mask[src] > 0
    return image, seg

def sample_views(volume, mask, config=None, training=False, rng=None):
    config = config or SamplingConfig()
    if volume.ndim != 4 or volume.shape[-1] != 2 or volume.shape[:3] != mask.shape:
        raise ValueError('Expected image [H,W,D,2] and matching 3D lesion mask')
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        raise ValueError('Empty lesion mask: anatomical boundary views cannot be located')
    rng = rng if rng is not None else np.random
    centroid = coords.mean(0)
    low, high = coords.min(0), coords.max(0) + 1
    main_start = np.floor(centroid).astype(int) - 64
    main_start = np.where(np.asarray(mask.shape) >= 128,
                          np.clip(main_start, 0, np.maximum(np.asarray(mask.shape)-128, 0)), main_start)
    plane_axes = [a for a in range(3) if a != config.slab_axis]
    margin_lo = config.extra_width
    context_start = low.copy()
    context_size = high - low
    for a in plane_axes:
        context_start[a] -= margin_lo
        context_size[a] += 2 * config.extra_width
    context_size[config.slab_axis] = config.thickness
    lower, upper = context_start.copy(), context_start.copy()
    lower[config.slab_axis] = low[config.slab_axis] - (config.thickness - config.overlap)
    upper[config.slab_axis] = high[config.slab_axis] - config.overlap
    center = np.floor(centroid).astype(int)
    if training and config.texture_jitter:
        draw = rng.integers if hasattr(rng, 'integers') else rng.randint
        offset = draw(-config.texture_jitter, config.texture_jitter+1, size=3)
        candidate = center + offset
        center = coords[np.argmin(np.sum((coords-candidate)**2, axis=1))]
    else:
        center = coords[np.argmin(np.sum((coords-centroid)**2, axis=1))]
    specs = [(main_start, np.array([128]*3)), (lower, context_size),
             (upper, context_size), (center-16, np.array([32]*3))]
    views, metadata = [], []
    for name, (start, size) in zip(VIEW_NAMES, specs):
        image, seg = crop_padded(volume, mask, start, size)
        native_shape = list(seg.shape)
        if name.endswith('context'):
            image = np.moveaxis(image, config.slab_axis, 2)
            seg = np.moveaxis(seg, config.slab_axis, 2)
            scale = min(config.plane_size/image.shape[0], config.plane_size/image.shape[1])
            factors = (scale, scale, 1)
            image = zoom(image, factors+(1,), order=1, prefilter=False)
            seg = zoom(seg, factors, order=0, prefilter=False)
            padding = [(0,0)]*3
            for a in (0,1):
                gap = config.plane_size - seg.shape[a]
                padding[a] = (gap//2, gap-gap//2)
            image = np.pad(image, padding+[(0,0)])
            seg = np.pad(seg, padding)
        views.append((np.ascontiguousarray(image), np.ascontiguousarray(seg)))
        metadata.append(dict(name=name, start=np.asarray(start).tolist(),
                             native_shape=native_shape, network_shape=list(seg.shape)))
    return views, dict(centroid=centroid.tolist(), bbox_start=low.tolist(),
                       bbox_end_exclusive=high.tolist(), config=config.to_dict(), views=metadata)
