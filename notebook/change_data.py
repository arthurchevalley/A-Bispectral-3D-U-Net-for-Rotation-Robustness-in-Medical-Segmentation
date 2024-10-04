import re


import numpy as np
from tqdm import tqdm
import glob
import SimpleITK as sitk
import json
import os

def make_isotropic(image, interpolator = sitk.sitkLinear, spacing = None):
    '''
    Many file formats (e.g. jpg, png,...) expect the pixels to be isotropic, same
    spacing for all axes. Saving non-isotropic data in these formats will result in
    distorted images. This function makes an image isotropic via resampling, if needed.
    Args:
        image (SimpleITK.Image): Input image.
        interpolator: By default the function uses a linear interpolator. For
                      label images one should use the sitkNearestNeighbor interpolator
                      so as not to introduce non-existant labels.
        spacing (float): Desired spacing. If none given then use the smallest spacing from
                         the original image.
    Returns:
        SimpleITK.Image with isotropic spacing which occupies the same region in space as
        the input image.
    '''
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    if spacing is None:
        spacing = min(original_spacing)
    if type(spacing) is list:
        new_spacing = []
        new_size = []
        for dim, osz, ospc in zip(range(3), original_size, original_spacing):
            if spacing[dim] is not None:
                new_size.append(int(round(osz*ospc/spacing[dim])))  
                new_spacing.append(spacing[dim])
            else:
                new_size.append(int(round(osz*ospc/original_spacing[dim])))
                new_spacing.append(original_spacing[dim])
    else:
        new_spacing = [spacing]*image.GetDimension()
        new_size = [int(round(osz*ospc/spacing)) for osz, ospc in zip(original_size, original_spacing)]
    return sitk.Resample(image, new_size, sitk.Transform(), interpolator,
                         image.GetOrigin(), new_spacing, image.GetDirection(), 0, # default pixel value
                         image.GetPixelID())
    
def downsamplePatient(original_CT, resize_factor):
    """
    Downsample a CT by a given resize factor
    """
    dimension = original_CT.GetDimension()
    reference_physical_size = np.zeros(original_CT.GetDimension())
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]
    
    reference_origin = original_CT.GetOrigin()
    reference_direction = original_CT.GetDirection()

    reference_size = [round(sz/resize_factor) for sz in original_CT.GetSize()] 
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(original_CT.GetDirection())

    transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)
  
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))

    
    centered_transform = sitk.CompositeTransform([transform, centering_transform])

    return sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0)
    
def fit_limit_volume(xlim,img_shape, limit_volume, RoI_dif_sizes):
    """
    Fits a volume to a given volume
    
    Input:
        xlim: the desired limit
        img_shape: input image shape
        limit_volume: maximum volume allowed
        RoI_dif_sizes: maximum non zero volume
    Return:
        xlim: the new limit
        success: Defines if the process was sucessful
    """
    success = True
    # non zero is bigger than limit size along x
    if RoI_dif_sizes > 0:
        xlim[0] = np.random.choice([i for i in range(xlim[0], xlim[1]-limit_volume)])
        xlim[1] = xlim[0] + limit_volume
    
    # non zero is smaller than limit size along x
    elif RoI_dif_sizes < 0:
        # if smaller RoI have to increase its size
        # best option an equal amount on all axis
        diff = abs(RoI_dif_sizes)

        if xlim[0] >= int(np.floor(diff/2)) and xlim[1]+(diff-int(np.floor(diff/2))) <= img_shape:
            xlim[0] = xlim[0] - int(np.floor(diff/2))
            xlim[1] = xlim[1] + diff-int(np.floor(diff/2))
        elif xlim[0] >= int(np.floor(diff/2)):
            # maximum value is up to the image border
            if (xlim[0]-(diff - (img_shape - xlim[1]))) >= 0:
                xlim[0] = xlim[0] - (diff - (img_shape - xlim[1]))
                xlim[1] = img_shape#xlim[1] - (img_shape)
            else:
                success = False
            
        elif xlim[0] < int(np.floor(diff/2)):
            # the new limit is the border, i.e. 0
            
            if (xlim[1]+diff-xlim[0]) <= img_shape:
                xlim[1] = xlim[1]+diff-xlim[0]
                xlim[0] = 0
            else:
                success = False
        else:
            success = False
            
    return xlim, success


def change_image_ATM(json_path, 
                     base_img_path, 
                     trg_img_path, 
                     limit_volume = None,
                     delta_spacing = 0.5, 
                     save = True, 
                     limit = 10000,
                     spacing = [2.,2.,2.],
                     data_train = [True,False]):

    """
    Convert ATM data to the desired output
    
    Input:
        json_path: str: json file path for the dataset
        base_img_path: str: path of the dataset
        trg_img_path: str: desired saving path
        limit_volume: maximum allowed volume if desired
        delta_spacing: float: maximum spacing allowed in any direction of the input data. Don't use data where the spacing is higher
        save: bool: Decide to save the processed data or not
        limit: int: Maximum number of volumes to proceed
        spacing: list: Desired spacing in each directions when making the data isoctropic
        data_train: Decide to either use the training data or testing data
        
    """
    labelsTr_path = "labelsTr"
    
    

    base_id = 704
    skipped = []
    training = []
    testing = []
    spacing_list = []
    size_list = []
    nz_size_list = []
    # sitk.sitkNearestNeighbor, sitk.sitkBSpline sitk.sitkLinear
    for train in data_train:
        if train:
            imagesTr_path = "imagesTr"
        else:
            imagesTr_path = "imagesTs"
            
        base_img_path_Tr = os.path.join(base_img_path, imagesTr_path)
        interpolator_img = sitk.sitkBSpline
        interpolator_lbl = sitk.sitkNearestNeighbor
        i = 0
        for image_path in tqdm(glob.glob(os.path.join(base_img_path_Tr,'*.nii.gz'))):
            i += 1
            if i > limit:
                break
            if train:
                label_path = re.sub(imagesTr_path, labelsTr_path, image_path)

            image_sitk = sitk.ReadImage(str(image_path))
            space = list(image_sitk.GetSpacing())

            size_list.append(list(image_sitk.GetSize()))
            spacing_list.append(space)
            if abs(space[0]-space[2])>delta_spacing and delta_spacing and (abs(1.-space[0])>delta_spacing or abs(1.-space[2])>delta_spacing):
                skipped.append(image_path)
                continue
            
            if train:
                label_sitk = sitk.ReadImage(str(label_path))
            
            resampled_img_crop = make_isotropic(image_sitk, spacing=spacing, interpolator=interpolator_img)
            resampled_img_crop.SetOrigin((resampled_img_crop.GetDirection()[0],resampled_img_crop.GetDirection()[4], resampled_img_crop.GetDirection()[-1]))
            if train:
                resampled_lbl_crop = make_isotropic(label_sitk, spacing=spacing, interpolator=interpolator_lbl)
                resampled_lbl_crop.SetOrigin((resampled_lbl_crop.GetDirection()[0],resampled_lbl_crop.GetDirection()[4], resampled_lbl_crop.GetDirection()[-1]))

                if len(np.unique(sitk.GetArrayFromImage(resampled_lbl_crop))) > 2:
                    skipped.append(image_path)
                    continue
            if limit_volume is not None:
                
                res_img_show = np.moveaxis(sitk.GetArrayFromImage(resampled_img), 0,2)    
                if train:
                    res_lbl_show = np.moveaxis(sitk.GetArrayFromImage(resampled_lbl), 0,2)
            
                if( (np.array(list(res_img_show.shape))-limit_volume)<0).any():
                    skipped.append(image_path)
                    continue
                
                if train:
                    res_lbl_show[res_lbl_show==-1] = 0
                    x, y, z = np.nonzero(res_lbl_show)
                    xlim = [x.min(), x.max()]
                    ylim = [y.min(), y.max()]
                    zlim = [z.min(), z.max()]
                else:
                    x = np.random.randint(0,max(res_img_show.shape[0]-limit_volume[0],1))
                    xlim = [x, x+limit_volume[0]]
                    y = np.random.randint(0,max(res_img_show.shape[1]-limit_volume[1],1))
                    ylim = [y, y+limit_volume[1]]
                    z = np.random.randint(0,max(res_img_show.shape[2]-limit_volume[2],1))
                    zlim = [z, z+limit_volume[2]]
                    
                RoI_dif_sizes = [(xlim[1]-xlim[0])-limit_volume[0],
                                (ylim[1]-ylim[0])-limit_volume[1],
                                (zlim[1]-zlim[0])-limit_volume[2]]

                xlim, success = fit_limit_volume(xlim, res_img_show.shape[0], limit_volume[0], RoI_dif_sizes[0])
                success = True
                if not success:
                    print(f'Image {image_path} does not meet the size criteria along x')
                    break
                ylim, success = fit_limit_volume(ylim, res_img_show.shape[1], limit_volume[1], RoI_dif_sizes[1])
                if not success:
                    print(f'Image {image_path} does not meet the size criteria along y')
                    break

                zlim, success = fit_limit_volume(zlim, res_img_show.shape[2], limit_volume[2], RoI_dif_sizes[2])
                if not success:
                    print(f'Image {image_path} does not meet the size criteria along z')
                    break
                    # otherwise where we can
                    
                resampled_img = resampled_img[ylim[0]:ylim[1], xlim[0]:xlim[1], zlim[0]:zlim[1]]
                if list(resampled_img.GetSize()) != limit_volume.tolist():
                    print(f'Save img {image_path[-10:-7]} size: {resampled_img.GetSize()} spacing{resampled_img.GetSpacing()}')
                if train:
                    resampled_lbl = resampled_lbl[ylim[0]:ylim[1], xlim[0]:xlim[1], zlim[0]:zlim[1]]
                    if list(resampled_lbl.GetSize()) != limit_volume.tolist():
                        print(f'Save lbl {label_path[-10:-7]} size: {resampled_lbl.GetSize()} spacing{resampled_lbl.GetSpacing()}')
            
            img_path = re.sub(base_img_path, trg_img_path, image_path)
            if save: 
                sitk.WriteImage(resampled_img_crop, img_path)

            if train:
                lbl_path = re.sub(base_img_path, trg_img_path, label_path)
                lbl_path_r = re.sub(re.sub(base_img_path_Tr, "", image_path)[4:7], str(base_id), label_path) 
                if save: 
                    sitk.WriteImage(resampled_lbl_crop, lbl_path)

            base_id += 1
            if train:
                training.append({'image':"./"+re.sub(base_img_path, "", image_path),'label':"./"+re.sub(base_img_path, "", label_path)})

            else:
                testing.append("./"+re.sub(base_img_path, "", image_path))

    with open(json_path) as f:
        data = json.load(f)
        
    data['numTraining'] = len(training)
    data['training'] = training
    
    data['numTest'] = len(testing)
    data['test'] = testing
    
    data['skipped_all'] = skipped
    data['spacing_list'] = spacing_list
    data['size_list'] = size_list
    data['nz_size_list'] = nz_size_list
    json_object = json.dumps(data, indent=4)
 
    # Writing to sample.json
    with open(re.sub(base_img_path, trg_img_path, json_path), "w") as outfile:
        outfile.write(json_object)
    
      
    
if __name__ == "__main__":
    base_path = os.path.join(os.getcwd(), 'src', 'data', 'raw', 'nnUNet_raw_data')
    json_path = os.path.join(base_path, 'ATM','dataset.json')
    base_img_path = os.path.join(base_path, 'ATM') 
    trg_img_path = os.path.join(base_path, 'ATM_clean')
    change_image_ATM(json_path=json_path, base_img_path=base_img_path, trg_img_path=trg_img_path) 