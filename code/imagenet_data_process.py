from PIL import Image

import os
import os.path
import torch
import torch.utils.data as data
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np
from torchvision import datasets, transforms, utils
import pdb

class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        return ""


class StandardTransform(object):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)



def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    Args:
        directory (str): root dataset directory
        class_to_idx (Dict[str, int]): dictionary mapping class name to class index
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.

    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/[...]/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/[...]/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class IMAGENET_DATA_PROCESS(DatasetFolder):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            seg_num=None,
            seg_idex=None,
            forward=False,
            head=False,
            tail=False

    ):
        super(IMAGENET_DATA_PROCESS, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.resize = transforms.Resize((224,224))
        self.seg_num = seg_num
        self.seg_idex = seg_idex

        self.forward = forward

        self.head = head
        self.tail = tail
        # pdb.set_trace()


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.resize(sample)
        # print(sample)
        # pdb.set_trace()
        # sample.save('/home2/pengyifan/pyf/freq/temp/before.png') #test
        # print(sample.size) #test
        sample = np.asarray(sample) ##########################################################yao
        # sample = Image.fromarray(sample)
        if self.transform is not None:

            h,w = sample.shape[:2]
            bsf = np.ones((h,w,3))

            if self.seg_num==3:
                radius = [0,37,74] 
                if self.seg_idex==self.seg_num:
                    R = 112 
                    r = 74
                else:
                    R = radius[self.seg_idex]
                    r = R-37

            if self.seg_num==4:
                radius = [0,28,56,84] 
                if self.seg_idex==self.seg_num:
                    R = 112 
                    r = 84
                else:
                    R = radius[self.seg_idex]
                    r = R-28        

            if self.seg_num==8:
                radius = [0,14,28,42,56,70,84,98] 
                if self.seg_idex==self.seg_num:
                    R = 112 
                    r = 98
                else:
                    R = radius[self.seg_idex]
                    r = R-14  

            if self.seg_num==16:
                radius = [0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105] 
                if self.seg_idex==self.seg_num:
                    R = 112 
                    r = 105
                else:
                    R = radius[self.seg_idex]
                    r = R-7 

            if self.seg_num==28:
                radius = [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108] 
                if self.seg_idex==self.seg_num:
                    R = 112 
                    r = 108
                else:
                    R = radius[self.seg_idex]
                    r = R-4            

            if self.seg_num==56:
                radius = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110] 
                if self.seg_idex==self.seg_num:
                    R = 112 
                    r = 110
                else:
                    R = radius[self.seg_idex]
                    r = R-2    

            if self.seg_num==100: #test
                bsf[56,45,:] = 1
                
            if self.seg_num!=100:
                for x in range(w):
                    for y in range(h):
                        if self.forward==False:
                            if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) <= (R**2) and ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2):
                                bsf[y,x,:] = 0 #Band stop filter
                        else:
                            if self.head and not self.tail:
                                if not (((x-(w-1)/2)**2 + (y-(h-1)/2)**2) <= (R**2) and ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2)):
                                    bsf[y,x,:] = 0 #Band stop filter
                            elif self.tail and not self.head:
                                if not (((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2)): #去尾（整个高频）
                                    bsf[y,x,:] = 0 #Band stop filter
                            else:
                                print("error!")

            
            freq = np.fft.fft2(sample,axes=(0,1))
            freq = np.fft.fftshift(freq)
            # (Image.fromarray(np.uint8(bsf*255))).save('/home2/pengyifan/pyf/freq/temp/bsf.png') #test
            # (Image.fromarray(np.uint8(freq))).save('/home2/pengyifan/pyf/freq/temp/freq.png') #test
            f = freq * bsf
            # (Image.fromarray(np.uint8(f))).save('/home2/pengyifan/pyf/freq/temp/f.png') #test
            sample = np.abs(np.fft.ifft2(np.fft.ifftshift(f),axes=(0,1)))
            sample = np.clip(sample,0,255)
            sample = sample.astype('uint8')    
            # (Image.fromarray(np.uint8(sample))).save('/home2/pengyifan/pyf/freq/temp/after.png') #test
            # pdb.set_trace()


            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target