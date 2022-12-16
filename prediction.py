from multiprocessing import Pool, Manager
from pathlib import Path
from typing import Generator, List, Tuple

import cv2
import numpy as np
from PIL import Image
import h5py as h5
from tqdm import tqdm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import dirwalk


def get_predictions(root_path, model_path, batch_size=8, nproc: int = -1, backend: str = 'DeepDanbooru'):
    """Get predictions for all images in a directory. Use cache if available.

    Args:
        root_path (pathlike): Path to directory containing images.
        model_path (pathlike): Path to model file.
        batch_size (int, optional): Batch size for prediction. Defaults to 8.
        nproc (int, optional): Number of processes to use in the dataloader. Defaults to -1 to use all available.
        backend (str, optional): Backend to use. Defaults to 'DeepDanbooru'. Also available: 'ConvNext'

    Returns:
        Tuple of (raw_predictions, paths)
    """
    img_paths = list(str(p.relative_to(root_path)) for p in dirwalk(Path(root_path)) if p.suffix.lower() in (".jpg", ".png", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp"))
    if os.path.exists(root_path + f'/prediction_cache_{backend}.h5'):
        print('Loading predictions from cache')
        with h5.File(root_path + f'/prediction_cache_{backend}.h5', 'r') as f:
            _cached_paths: np.ndarray = f['paths'][:]  # type: ignore
            cached_preds: np.ndarray = f['preds'][:]  # type: ignore
        cached_paths = [str(p, encoding='utf8') for p in _cached_paths]
        img_paths_set = set(img_paths)
        mask = np.array([p in img_paths_set for p in cached_paths])
        cached_paths = np.array(cached_paths)[mask].tolist()
        cached_preds = cached_preds[mask]
        new_paths = [p for p in img_paths if p not in set(cached_paths)]
        print(f'Found {len(new_paths)} new images, {len(img_paths)-len(new_paths)} already cached')
    else:
        print(f'No cache found. Predicting all {len(img_paths)} images.')
        new_paths = img_paths
        cached_paths = []
        cached_preds = None  # type: ignore

    if new_paths:
        new_paths = [Path(root_path) / p for p in new_paths]
        predictor = Predictor(model_path=model_path, root_path=root_path, img_paths=new_paths, nproc=nproc, backend=backend)
        p, a = predictor.predict(batch_size=batch_size)
        sp = [str(x.relative_to(root_path)) for x in p]
        a = a.astype(np.float16)
        if cached_preds is None:
            preds = a
            paths = sp
        else:
            preds = np.concatenate((cached_preds, a), axis=0)
            paths = cached_paths + sp
        print('Saving cache...')
        with h5.File(root_path + f'/prediction_cache_{backend}.h5', 'w') as f:
            f.create_dataset('paths', data=np.array(paths, dtype=h5.special_dtype(vlen=str)), compression='gzip', compression_opts=9)
            f.create_dataset('preds', data=preds, compression='gzip', compression_opts=9)
        print('Done.')
    else:
        preds = cached_preds
        paths = cached_paths
    return preds, paths


def load_and_process_image(image, target_width, target_height, backend):
    """
    Transform image and pad by edge pixles.
    Args:
        image (PIL.Image.Image | Path like object): image to transform
        target_width (int): target width
        target_height (int): target height
    Returns:
        np.ndarray: transformed image    
    """
    p = image
    if not isinstance(image, np.ndarray):
        try:
            if isinstance(image, str) or isinstance(image, Path):
                pil_img = Image.open(image)
            else:
                assert isinstance(image, Image.Image)
                pil_img = image

            if pil_img.mode not in ("RGB", "RGBA"):
                pil_img = pil_img.convert("RGBA")
            image = np.array(pil_img)
            del pil_img
            assert isinstance(image, np.ndarray)
        except Exception as e:
            with open("/tmp/fast_deepdanbooru_failed.txt", "a") as f:
                f.write(str(p) + "\n")
            print(f"Failed to open {p}")
            print(e)
            return None

    y, x = image.shape[:2]
    X, Y = target_width, target_height
    nx, ny = min(X, int(X * x / y)), min(Y, int(Y * y / x))
    image = cv2.resize(image, (nx, ny), interpolation=cv2.INTER_AREA)
    dx, dy = (X - nx) // 2, (Y - ny) // 2
    if backend == 'DeepDanbooru':
        image = cv2.copyMakeBorder(image, dy, Y - ny - dy, dx, X - nx - dx, cv2.BORDER_REPLICATE)
    else:
        image = cv2.copyMakeBorder(image, dy, Y - ny - dy, dx, X - nx - dx, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if image.shape[2] == 4:
        RGB, alpha = image[:, :, :3], image[:, :, 3:]
        alpha = alpha / 255
        RGB[:] = (1 - alpha) + alpha * RGB
        image = RGB
    if backend == 'DeepDanbooru':
        return image.astype(np.float32) / 255
    else:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def worker(x):
    """Worker function for dataloader. This function is called by multiprocessing.Pool.map_async to load images in parallel and do transformation.

    Args:
        x (tuple): (image path, target width, target height)
    """
    q, path, backend, H, W = x
    img = load_and_process_image(path, H, W, backend)
    if img is not None:
        q.put((path, img))


class DataLoader:
    def __init__(self, root_path=None, img_paths=None, nproc: int = 8, maximum_look_ahead: int = 128, backend='DeepDanbooru'):
        """Multi-threaded image dataloader for deepdanbooru. All necessary preprocesses are done in this function.

        Args:
            root_path (Path): Path to the root directory of images. If img_paths is not specified, this argument is required.
            img_paths (List[Path]): List of image paths. If this is not None, root_path is ignored.
            nproc (int, optional): Number of processes for image loading. Defaults to 8.
            maximum_look_ahead (int, optional): Maximum number of images to preload. Defaults to 128.
            backend (str, optional): Backend to use. Defaults to 'DeepDanbooru'. Also available: ConvNext
        """
        self.root = root_path
        self.nproc = nproc if nproc > 0 else None
        self.maximum_look_ahead = maximum_look_ahead
        self.backend=backend
        if img_paths is None:
            assert root_path is not None
            self.img_paths = list(p for p in dirwalk(Path(root_path)) if p.suffix.lower() in (".jpg", ".png", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp"))
        else:
            self.img_paths = img_paths
        self.generator = self.image_generator(self.nproc, self.maximum_look_ahead, backend=self.backend)

    def __call__(self):
        return self.generator

    def image_generator(self, nproc, maximum_look_ahead: int = 128, backend='DeepDanbooru'):
        """Multi-threaded image dataloader for deepdanbooru. All necessary preprocesses are done in this function.

        Args:
            nproc (int): number of processes. Defaults to 8.
            maximum_look_ahead (int): maximum number of images preload. Defaults to 128.

        Yields:
            tuple: (image path, image)
        """
        path = self.img_paths
        pool = Pool(nproc)
        manager = Manager()
        H, W = (512, 512) if backend == 'DeepDanbooru' else (448, 448)
        q = manager.Queue(maxsize=maximum_look_ahead)
        

        def gen(q, imgs):
            for img in imgs:
                yield q, img, backend, H, W

        r = pool.map_async(worker, gen(q, path))

        for _ in range(len(path)):
            yield q.get()


class Predictor:
    def __init__(self, model_path, root_path=None, img_paths=None, nproc: int = -1, backend='DeepDanbooru'):
        """Run the prediction on a directory of images

        Args:
            root_path (pathlike): Path to the root directory of images. If img_paths is not specified, this argument is required.
            model_path (pathlike): Path to the keras model.
            nproc (int, optional): Number of processes for image loading. Defaults to all available cores.
            backend (str, optional): Backend to use. Defaults to 'DeepDanbooru'. Available backends are 'DeepDanbooru' and 'ConvNext'.
        """
        self.root_path = root_path
        self.model_path = Path(model_path)
        self.nproc = nproc
        self.model = None  # type: ignore
        self.backend = backend

        if img_paths is None:
            assert root_path is not None
            self.img_paths = list(p for p in dirwalk(Path(root_path)) if p.suffix.lower() in (".jpg", ".png", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp"))
        else:
            self.img_paths = img_paths

    def load_model(self):
        import tensorflow as tf
        # Suppress tensorflow warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        model: tf.keras.Model = tf.keras.models.load_model(self.model_path)  # type: ignore
        assert model is not None, "Model is not loaded, please check the model path"
        # skip sigmoid layer
        self.model: tf.keras.Model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    def predict(self, batch_size: int = 8) -> Tuple[List[Path], np.ndarray]:
        """Load tf model and predict on all images registered in the dataloader. Notice that this function will load all predictions into memory, and the returned path order is generally not the same as the input path order due to the multi-threading.

        Args:
            batch_size (int, optional): Defaults to 8.

        Returns:
            Tuple[List[Path], np.ndarray]: (image paths, predictions)
        """

        import tensorflow as tf
        # set mompry growth to true
        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        if not gpu_devices:
            print("No GPU found, using CPU. Number of workers will be set to 1")
            self.nproc = 1
        else:
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
        self.dataloader = DataLoader(root_path=self.root_path, img_paths=self.img_paths, nproc=self.nproc, backend=self.backend)
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

        self.load_model() if self.model is None else None
        paths = []

        def gen():
            for path, arr in self.dataloader():
                paths.append(path)
                yield arr
        shape = (512, 512, 3) if self.backend == 'DeepDanbooru' else (448, 448, 3)
        ds = tf.data.Dataset.from_generator(gen, tf.float32, tf.TensorShape(shape))
        print("Initializing...")

        class KerasPbar(tf.keras.callbacks.Callback):
            def __init__(self, total, batch_size, **tqdm_params):
                super().__init__()
                self.batch_size = batch_size
                self.tqdm_params = tqdm_params
                self.total = total

            def on_predict_batch_end(self, batch, logs=None):
                remaining = self.total - (batch + 1) * self.batch_size
                N = int(self.batch_size * (batch - self.prev_predict_batch))
                self.tqdm_progress.update(min(remaining, N))
                self.prev_predict_batch = batch

            def on_predict_begin(self, logs=None):
                self.prev_predict_batch = 0
                self.tqdm_progress = tqdm(total=self.total, **self.tqdm_params)

            def on_predict_end(self, logs=None):
                remaining = self.total - self.tqdm_progress.n
                self.tqdm_progress.update(remaining)

        keras_pbar = KerasPbar(len(self.img_paths), batch_size, desc="Predicting", unit="image")
        ds = ds.batch(batch_size)
        r = self.model.predict(ds, callbacks=[keras_pbar], verbose='0')
        return paths, r
