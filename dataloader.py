# %%
from torch.utils.data import Dataset
from torchvision import transforms
import useful_wsi as usi
from glob import glob
import numpy as np
import os
from xml.dom import minidom
import skimage
from skimage.draw import polygon
import pandas as pd
##
# Garder en tete que l'attribution 0/1 à la target est aléatoire. De même dans le dataset MIL ??
##


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

#ask_level = 3
#path_xml = 'data_test/image_tcga_2.xml'
#path_wsi = 'data_test/image_tcga_2.svs'
#def mask_funcion(image):
#    _ = image
#    mask = get_polygon(path_xml=path_xml, label='t')
#    return mask
#slide = usi.utils.open_image(path_wsi)
#para = usi.patch_sampling(slide = path_wsi, mask_level=3, mask_function=mask_funcion, sampling_method='random_patches', n_samples=4)
#im = usi.get_image(path_wsi, para[0], numpy=False)
#%%
#def get_patches(path_wsi, path_xml, patches_per_wsi):


#%%
class fromWsi(Dataset):
    def __init__(self, path_wsi, path_xml, n_patches_per_wsi, label_xml, table_data, target_name='LST_status'):

        self.n_patches_per_wsi = n_patches_per_wsi
        table_data = pd.read_csv(table_data)
        self.table_data = table_data
        self.label_xml = label_xml
        self.path_xml = path_xml
        files = np.array(glob(path_wsi + '/*.svs'))
        files = files[self.is_in_table(files)]
        self.target_name = target_name
        self.transform_target()
        para_patches = []
        targets = []
        for f in files:
            para = self.extract_patches(f=f)
            targets += self.extract_target(f)
            para_patches += para

        self.para_patches = tuple(para_patches)
        self.targets = tuple(targets)
        self.transform =  transforms.Compose([transforms.ToTensor()])

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
        para = usi.patch_sampling(slide = f, mask_level=3, mask_function=mask_function, 
                                  sampling_method='random_patches', n_samples=self.n_patches_per_wsi)
        s = [f]*len(para)
        out += list(zip(s, para))       
        return out

    def transform_target(self):
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

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        slide, para = self.para_patches[idx]
        image = usi.utils.get_image(slide=slide, para=para, numpy=False)
        image = image.convert('RGB')
        sample = {'image': image,
                  'target': self.targets[idx]}
        return sample

ds = fromWsi(path_wsi='data_test/image', path_xml='data_test/annot', 
            n_patches_per_wsi=10, label_xml='t', table_data='data_test/labels_tcga_tnbc.csv')
        
print(ds[0])
