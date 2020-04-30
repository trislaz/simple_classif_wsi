# %%
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from PIL import Image
from torchvision import transforms 
import useful_wsi as usi
from glob import glob
import numpy as np
import os
from xml.dom import minidom
import skimage
from skimage.draw import polygon
import pandas as pd
#%%

##
# Garder en tete que l'attribution 0/1 à la target est aléatoire. Parfaitement symétrique dans ce cas ! De même dans le dataset MIL ??
##

# Use 
def make_loaders(args):
    dataset_train = datasetWSI_simple(args=args, transform=get_transform(train=True, color_aug=args.color_aug))
    dataset_val = datasetWSI_simple(args=args, transform=get_transform(train=False))
    indices = list(range(len(dataset_train)))
    split = len(dataset_train) // 3

    # Shuffles dataset
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    val_indices, train_indices = indices[:split], indices[split:]
    val_sampler = SubsetRandomSampler(val_indices)
    train_sampler = SubsetRandomSampler(train_indices)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=24)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, sampler=val_sampler, num_workers=24)
    return dataloader_train, dataloader_val


def get_polygon(path_xml, label):
    
    doc = minidom.parse(path_xml).childNodes[0]
    nrows = doc.getElementsByTagName('imagesize')[0].getElementsByTagName('nrows')[0].firstChild.data
    ncols = doc.getElementsByTagName('imagesize')[0].getElementsByTagName('ncols')[0].firstChild.data
    size_image = (int(nrows), int(ncols))
    mask = np.zeros(size_image)
    obj = doc.getElementsByTagName('object')
    polygons = []
    for o in obj:
        if o.getElementsByTagName('name')[0].firstChild.data == label:
            polygons += o.getElementsByTagName('polygon')
            print(polygons)
    if not polygons:
        raise ValueError('There is no annotation with label {}'.format(label))

    for poly in polygons:
        rows = []
        cols = []
        for point in poly.getElementsByTagName('pt'):
            x = int(point.getElementsByTagName('x')[0].firstChild.data)
            y = int(point.getElementsByTagName('y')[0].firstChild.data)
            rows.append(y)
            cols.append(x)
        rr, cc = polygon(rows, cols)
        mask[rr, cc] = 1
    return mask

# To differentiate between train and test, maybe "build dataset and get transforms" manually, not in __init__.
class fromWsi(Dataset):
    def __init__(self, path_wsi, path_xml, n_patches_per_wsi, resolution, 
                 table_data, label_xml='t', target_name='LST_status', color_aug=0, train=True):

        self.resolution = resolution
        self.n_patches_per_wsi = n_patches_per_wsi
        self.train = train
        table_data = pd.read_csv(table_data)
        self.table_data = table_data
        self.label_xml = label_xml
        self.path_xml = path_xml
        files = np.array(glob(path_wsi + '/*.svs'))
        files = files[self.is_in_table(files)]
        files = files[self.has_masks(files)]
        self.files = files
        self.target_name = target_name
        self.color_aug = color_aug
        self.para_patches, self.targets = self.build_dataset()
        self.transform = None

    def get_transform(self):
        """Transforms the images to augment the dataset
        Default transformations are randomrotations and flip
        Other Transformations that can be added are color deformation and gray
        scale.
        
        Returns
        -------
        transforms.Compose
            pipeline of transformation to apply.
        """
        if self.train:
            if self.color_aug:
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    #transforms.RandomRotation(degrees=180),
                    transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3)], p=0.5),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor()
                ])
            else:
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    #transforms.RandomRotation(degrees=180),
                    transforms.ToTensor()
                ])

        else:
            transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform


    def build_dataset(self):
        self.transform_target()
        para_patches = []
        targets = []
        for f in self.files:
            para = self.extract_patches(f=f)
            targets += self.extract_target(f)
            para_patches += para
        return tuple(para_patches), tuple(targets)

    def extract_patches(self, f):
        """Extracts n patches of the wsi stocked at $path, with mask $xml.
        
        Parameters
        ----------
        f : str
            path to the wsi image
        xml : str
            path to the xml masks (the name of the masks must be th same than the images, 
            except for the extension)
        label : str
            label of the mask to use.
        n_patches_per_wsi: int
            number of patches to sample in each image
        
        Returns
        -------
        list
            list - each element is a tuple associating the path to the wsi to the parameters of the 
            extracted patches (these can feed the useful_wsi.get_image)
        """
        out = []
        name, _ = os.path.splitext(os.path.basename(f))
        xml = os.path.join(self.path_xml, name + '.xml')
        mask_function = lambda x: get_polygon(path_xml=xml, label=self.label_xml)
        para = usi.patch_sampling(slide = f, mask_level=3, mask_function=mask_function, mask_tolerance=0.2, patch_size=(256, 256),
                                  sampling_method='random_patches', n_samples=self.n_patches_per_wsi, analyse_level=self.resolution)
        s = [f]*len(para)
        out += list(zip(s, para))       
        return out

    def transform_target(self):
        """Adds to table to self.table_data
        a numerical encoding of the target. Works for classif.
        New columns is named "target"
        """
        table = self.table_data
        T = pd.factorize(table[self.target_name])
        table['target'] = T[0]
        self.target_correspondance = T[1]
        self.table_data = table

    def is_in_table(self, files):
        """Returns a boolean vectors indicating if a file is referenced in table_data.
        
        Parameters
        ----------
        files : list
            list of path to the image files
        table_data : pd.DataFrame
            dataframe of describing the data
        
        Returns
        -------
        list
            boolean vector
        """

        out = []
        valid_names = set(self.table_data['ID'].values)
        for f in files:
            name, _ = os.path.splitext(os.path.basename(f))
            out.append(name in valid_names)
        if not any(out):
            raise AttributeError("There is no image files or their names are not matching the names in the table_data.")
        return out

    def has_masks(self, files):
        out = []
        name_xmls = glob(os.path.join(self.path_xml, '*.xml'))
        name_xmls = [os.path.splitext(os.path.basename(x))[0] for x in name_xmls]
        for f in files:
            name_image, _ = os.path.splitext(os.path.basename(f))
            out.append(name_image in name_xmls)
            if not out:
                print('The mask of image {} is missing. Excluding it from database.'.format(name_image))
        return out

    def extract_target(self, f):
        """extracts targets of f from the table_data
        
        Parameters
        ----------
        f : str
            full path to an image
        
        Returns
        -------
        list
            target repeated $n_patches_per_wsi times
        """
        name, _ = os.path.splitext(os.path.basename(f))
        target = self.table_data[self.table_data['ID'] == name]['target'].values[0]
        target = [target] * self.n_patches_per_wsi
        return target

    def view_image(self, idx):
        image, _ = self.__getitem__(idx)
        image = transforms.ToPILImage()(image)
        return image
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        if self.transform is None:
            raise AttributeError('self.transform has not been assigned yet. Please set self.train and execute self.get_transform')
        slide, para = self.para_patches[idx]
        image = usi.utils.get_image(slide=slide, para=para, numpy=False)
        image = image.convert('RGB')
        image = self.transform(image)
        return image, self.targets[idx] 

#ds = fromWsi(path_wsi='/mnt/data4/tlazard/data/tcga_tnbc/images', path_xml='/mnt/data4/tlazard/data/tcga_tnbc/annotations/annotations_tcga_tnbc_guillaume', 
#           n_patches_per_wsi=10, label_xml='t', table_data='/mnt/data4/tlazard/data/tcga_tnbc/sample_labels.csv', resolution=0)           

class dataset(Dataset):
    imext = set(['.png', '.jpg'])
    def __init__(self, args, transform=None):
        self.path = args.wsi
        self.transform = transform
        self.target_name = args.target_name
        self.n_sample = args.n_sample
        self.table_data = pd.read_csv(args.table_data)
        self.target_dict = dict()
        self.imext = set(['.png', '.jpg'])
        self.files = self._collect_files(args.wsi)
        
    def _collect_file_path(self, path):
        """Creates a list of all path to the images.
        """
        out = []
        slide = os.path.basename(path)
        _ = self.make_target_dict(slide)
        files = os.listdir(path)
        files = [os.path.splitext(x) for x in files]
        for name, ext in files:
            if ext in self.imext:
                out.append((os.path.join(path, name+ext), path))  
        return out 

    def make_target_dict(self, slide):
        """extracts targets of f from the table_data
        
        Parameters
        ----------
        slide : str
            name of the slide
        """
        is_in_table = slide in set(self.table_data['ID'])
        if is_in_table:
            target = self.table_data[self.table_data['ID'] == slide]['target'].values[0]
            self.target_dict[slide] = target
        return is_in_table
    
    def _collect_files(self, path):
        self.transform_target()
        return self._collect_file_path(path)

    def __len__(self):
        return len(self.files)

    def transform_target(self):
        """Adds to table to self.table_data
        a numerical encoding of the target. Works for classif.
        New columns is named "target"
        """
        table = self.table_data
        T = pd.factorize(table[self.target_name])
        table['target'] = T[0]
        self.target_correspondance = T[1]
        self.table_data = table

    def __getitem__(self, idx):
        impath, slide_path = self.files[idx]
        image = Image.open(impath)
        name_slide = os.path.basename(slide_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.target_dict[name_slide]

class datasetWSI_simple(dataset):
    def __init__(self, args, transform=None):
        super(datasetWSI_simple, self).__init__(args, transform)

    def _collect_files(self, path):
        """Collects all files : path is a folder full of folders, each being a wsi.
        """
        out = []
        self.transform_target()
        paths = glob(os.path.join(path, '*'))
        for p in paths:
            is_in_table = self.make_target_dict(os.path.basename(p))
            if is_in_table:
                all_patches = self._collect_file_path(p)
                np.random.shuffle(all_patches)
                out += all_patches[:self.n_sample] #Randomly choose the first samples 
        return out

def get_transform(train, color_aug=False):
    if train:
        if color_aug:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3)], p=0.5),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor()])
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor()])
    else:
        transform = transforms.ToTensor()
    return transform
 
