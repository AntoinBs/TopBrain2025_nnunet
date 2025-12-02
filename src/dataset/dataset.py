import os
from glob import glob
import random
import logging
import shutil
from typing import Union

import numpy as np

import nibabel as nib
import SimpleITK as sitk
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

GLOBAL_LABELS = {
    0 : "background",
    1 : "BA",
    2 : "R-P1P2",
    3 : "L-P1P2",
    4 : "R-ICA",
    5 : "R-M1",
    6 : "L-ICA",
    7 : "L-M1",
    8 : "R-Pcom",
    9 : "L-Pcom",
    10 : "Acom",
    11 : "R-A1A2",
    12 : "L-A1A2",
    13 : "R-A3",
    14 : "L-A3",
    15 : "3rd-A2",
    16 : "3rd-A3",
    17 : "R-M2",
    18 : "R-M3",
    19 : "L-M2",
    20 : "L-M3",
    21 : "R-P3P4",
    22 : "L-P3P4",
    23 : "R-VA",
    24 : "L-VA",
    25 : "R-SCA",
    26 : "L-SCA",
    27 : "R-AICA",
    28 : "L-AICA",
    29 : "R-PICA",
    30 : "L-PICA",
    31 : "R-AChA",
    32 : "L-AChA",
    33 : "R-OA",
    34 : "L-OA",
    35 : "R-ECA",
    36 : "L-ECA",
    37 : "R-STA",
    38 : "L-STA",
    39 : "R-MaxA",
    40 : "L-MaxA",
    41 : "R-MMA",
    42 : "L-MMA",
    43 : "VoG",
    44 : "StS",
    45 : "ICVs",
    46 : "R-BVR",
    47 : "L-BVR",
    48 : "SSS"
}
CTA_TO_GLOBAL = {35 : 43, 36 : 44, 37 : 45, 38 : 46, 39 : 47, 40 : 48}
GLOBAL_TO_CTA = {v: k for k, v in CTA_TO_GLOBAL.items()}

class BiasFieldCorrection:
    """Apply N4 Bias Field Correction to a SimpleITK Image."""
    def __init__(self, shrink_factor: int = 1):
        self.shrink_factor = shrink_factor

    def __call__(self, img_sitk: sitk.Image) -> sitk.Image:
        # Create a mask using Otsu's method
        mask = sitk.OtsuThreshold(img_sitk, 0, 1, 200)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        # Set maximum number of iterations for each level
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])

        shrink_factor = self.shrink_factor
        if shrink_factor > 1:
            img_shrink = sitk.Shrink(img_sitk, [shrink_factor] * img_sitk.GetDimension())
            mask_shrink = sitk.Shrink(mask, [shrink_factor] * mask.GetDimension())

            # Apply N4 bias field correction to shrunken images
            corrector.Execute(img_shrink, mask_shrink)
            
            # Get the log bias field for original image
            log_bias_field = corrector.GetLogBiasFieldAsImage(img_sitk)
            # Get the corrected image from original image resolution
            img_corrected = img_sitk / sitk.Exp(log_bias_field)
        else:
            img_corrected = corrector.Execute(img_sitk, mask)

        return img_corrected

class LabelRemapping:
    """
    Remap label values in a SimpleITK segmentation image.
    You should use this only for CTA images.
    """
    def __init__(self, mapping: dict):
        """
        Args:
            mapping (dict): Dictionary mapping original label values to new label values.
                            Example: {35: 43, 36: 44}
        """
        self.mapping = mapping

    def __call__(self, label_sitk: sitk.Image) -> sitk.Image:
        """
        Apply the label remapping to the input segmentation.
        """
        change_filter = sitk.ChangeLabelImageFilter()
        
        # Ensure keys and values are standard python types for SimpleITK
        change_map = {float(k): float(v) for k, v in self.mapping.items()}
        change_filter.SetChangeMap(change_map)
        
        remapped_label = change_filter.Execute(label_sitk)
        
        return remapped_label

class IntensityClipping:
    """
    Clip the intensity values of a SimpleITK image to a specified range.
    You should use this only for CT images.
    """
    def __init__(self, min_val: float = -200, max_val: float = 800):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img_sitk: sitk.Image) -> sitk.Image:
        return sitk.Clamp(img_sitk, lowerBound=self.min_val, upperBound=self.max_val)
    
class PreprocessingPipeline:
    '''
    # Example usage:
    preprocessing_pipeline = PreprocessingPipeline('datasets/volumes', 'datasets/segmentations', 'datasets/new/volumes', 'datasets/new/segmentations', modality='CT' or 'MR')
    preprocessing_pipeline.run()
    '''
    def __init__(self, volume_dir, segmentation_dir, output_volumes_dir, output_segmentations_dir, modality=None, foreground_binarize=False):
        self.volume_dir = volume_dir
        self.segmentation_dir = segmentation_dir
        self.output_volumes_dir = output_volumes_dir
        self.output_segmentations_dir = output_segmentations_dir
        self.modality = modality
        self.foreground_binarize = foreground_binarize

    def run(self):
        # Ensure the output directories exist
        os.makedirs(self.output_volumes_dir, exist_ok=True)
        os.makedirs(self.output_segmentations_dir, exist_ok=True)
        
        # Get all NIfTI files in the volumes and segmentation directories
        volume_files = [f for f in os.listdir(self.volume_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        segmentation_files = [f for f in os.listdir(self.segmentation_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

        # Assuming filenames are the same for corresponding volume and segmentation
        for volume_file in volume_files:
            if volume_file in segmentation_files:
                volume_path = os.path.join(self.volume_dir, volume_file)
                segmentation_path = os.path.join(self.segmentation_dir, volume_file)

                try:
                    # Load the volume and segmentation
                    volume = sitk.ReadImage(volume_path)
                    segmentation = sitk.ReadImage(segmentation_path)
                    
                    # Reorient to RAS
                    desired_orientation = "RAS"
                    segmentation = sitk.DICOMOrient(segmentation, desired_orientation)
                    volume = sitk.DICOMOrient(volume, desired_orientation)

                    # Copy metadata from segmentation to volume
                    volume.SetOrigin(segmentation.GetOrigin())
                    volume.SetDirection(segmentation.GetDirection())
                    volume.SetSpacing(segmentation.GetSpacing())

                    if self.modality is not None:
                        # Additional preprocessing based on modality
                        if self.modality == 'CT':
                            clipper = IntensityClipping(min_val=-200, max_val=800) # Apply intensity clipping for CT images
                            volume = clipper(volume)
                            if not self.foreground_binarize:
                                mapper = LabelRemapping(CTA_TO_GLOBAL) # Apply label remapping for CTA segmentations
                                segmentation = mapper(segmentation)
                            else:
                                # Binarize the segmentation to foreground/background
                                segmentation = sitk.BinaryThreshold(segmentation, lowerThreshold=1, upperThreshold=1000, insideValue=1, outsideValue=0)


                        elif self.modality == 'MR':
                            bfc = BiasFieldCorrection(shrink_factor=4) # Apply bias field correction for MR images
                            volume = bfc(volume)
                            if self.foreground_binarize:
                                segmentation = sitk.BinaryThreshold(segmentation, lowerThreshold=1, upperThreshold=1000, insideValue=1, outsideValue=0)

                    # Save the modified volume in the output volumes directory
                    modified_volume_path = os.path.join(self.output_volumes_dir, volume_file)
                    sitk.WriteImage(volume, modified_volume_path)
                    print(f'Modified volume saved to: {modified_volume_path}')

                    # Save the segmentation in the output segmentations directory without changing the filename
                    modified_segmentation_path = os.path.join(self.output_segmentations_dir, volume_file)
                    sitk.WriteImage(segmentation, modified_segmentation_path)
                    print(f'Segmentation saved to: {modified_segmentation_path}')
                    
                except RuntimeError as e:
                    print(f"Skipping {volume_file} due to error: {e}")

            else:
                print(f"No matching segmentation found for volume: {volume_file}")

class DataSplitter:
    """
    Split dataset into training and test sets, based on label presence. 
    It means that the split will try to keep the same proportion of images having each label in the training and test sets.
    """
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def build_label_presence_matrix(self, segmentations_foder_path: str) -> np.ndarray:
        """
        Build a matrix indicating the labels presents and not in the segmentations located at segmentations_foder_path.
        (e.g. [(1, 0, 1, ...), (1, 1, 1, ...)] means :
            - image 1 : label 0 is present, label 1 is not present, label 2 is present, etc.)
            - image 2 : label 0 is present, label 1 is present, label 2 is present, etc.)
        """
        segmentation_files = [f for f in os.listdir(segmentations_foder_path) if f.endswith('.nii') or f.endswith('.nii.gz')]
        matrix = []
        for segmentation_file in segmentation_files:
            segmentation_path = os.path.join(segmentations_foder_path, segmentation_file)
            label_map = nib.load(segmentation_path).get_fdata()
            if label_map is not None:
                unique_labels = np.unique(label_map)
                presence_vector = np.zeros(49, dtype=int)  # Assuming labels range from 0 to 48
                for label in unique_labels:
                    presence_vector[int(label)] = 1
            else:
                presence_vector = np.zeros(49, dtype=int)  # No labels present
            matrix.append(presence_vector)

        return np.array(matrix)
    
    def split(self, data_path: Union[str, list], segmentation_path: Union[str, list], split_path: str):
        """
        Split datas in the folder(s) data and labels into training and test sets.
        If multiple folders are provided, they will be concatenated together, 
        and self.test_size of each folder will compose the test set.
        """
        if isinstance(data_path, str):
            data_path = [data_path]

        train_path = os.path.join(split_path, 'train')
        test_path = os.path.join(split_path, 'test')
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        os.makedirs(os.path.join(train_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(train_path, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(test_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(test_path, 'labels'), exist_ok=True)

        for dp, sp in zip(data_path, segmentation_path):
            print(f"Processing folder: {dp} and {sp}")
            volumes = [f for f in os.listdir(dp) if f.endswith('.nii') or f.endswith('.nii.gz')]
            segmentations = [f for f in os.listdir(sp) if f.endswith('.nii') or f.endswith('.nii.gz')]

            matrix = self.build_label_presence_matrix(sp)

            msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            train_indices, test_indices = next(msss.split(volumes, matrix))
            
            for train_idx in train_indices:
                if volumes[train_idx] in segmentations:
                    v_path = os.path.join(dp, volumes[train_idx])
                    s_path = os.path.join(sp, volumes[train_idx])
                    shutil.copy(v_path, os.path.join(train_path, 'images', volumes[train_idx]))
                    shutil.copy(s_path, os.path.join(train_path, 'labels', volumes[train_idx]))
                    print(f"Copied {volumes[train_idx]} to training set.")


            for test_idx in test_indices:
                if volumes[test_idx] in segmentations:
                    v_path = os.path.join(dp, volumes[test_idx])
                    s_path = os.path.join(sp, volumes[test_idx])
                    shutil.copy(v_path, os.path.join(test_path, 'images', volumes[test_idx]))
                    shutil.copy(s_path, os.path.join(test_path, 'labels', volumes[test_idx]))
                    print(f"Copied {volumes[test_idx]} to test set.")

class DataRenamer:
    """
    If we use the pycad splitter to create the train/valid/test folders, then this class is adapted for that, and is waiting for the folders train and valid with the subforlders images and labels.
    """

    def __init__(self, path_to_input, path_to_output, dataset_id, structure):
        self.dataset_id = dataset_id
        self.structure = structure

        self.path_to_train_image = glob(os.path.join(path_to_input, "train/images/*.nii.gz"))
        self.path_to_train_labels = glob(os.path.join(path_to_input, "train/labels/*.nii.gz"))
        self.path_to_test_image  = glob(os.path.join(path_to_input, "test/images/*.nii.gz"))
        self.path_to_test_labels  = glob(os.path.join(path_to_input, "test/labels/*.nii.gz"))


        output_path  = f"{path_to_output}/Dataset{self.dataset_id}_{self.structure}"
        self.path_to_nnunet_imagesTr = os.path.join(output_path, "imagesTr")
        self.path_to_nnunet_labelsTr = os.path.join(output_path, "labelsTr")
        self.path_to_nnunet_imagesTs = os.path.join(output_path, "imagesTs")

        os.makedirs(self.path_to_nnunet_imagesTr, exist_ok=True)
        os.makedirs(self.path_to_nnunet_imagesTs, exist_ok=True)
        os.makedirs(self.path_to_nnunet_labelsTr, exist_ok=True)
    
    def rename_train_data(self):
        for i, (vol, seg) in enumerate(zip(self.path_to_train_image, self.path_to_train_labels)):

            # Rename the training segmentations
            print(f"Segmentation file: {seg}")
            new_seg_filename = f"{self.structure}_{str(i).zfill(3)}.nii.gz"
            new_seg_filepath = os.path.join(self.path_to_nnunet_labelsTr, new_seg_filename) 
            print(f"new segmenation file: {new_seg_filepath}")

            shutil.copy(seg, new_seg_filepath)

            # Rename the training volumes
            print(f"Volume file: {vol}")
            new_volume_filename = f"{self.structure}_{str(i).zfill(3)}_0000.nii.gz"
            new_volume_filepath = os.path.join(self.path_to_nnunet_imagesTr, new_volume_filename)
            print(f"new volume file: {new_volume_filepath}") 

            shutil.copy(vol, new_volume_filepath)
    
    def rename_test_data(self):
        for i, (vol, seg) in enumerate(zip(self.path_to_test_image, self.path_to_test_labels)):

            # Rename the testing volumes
            print(f"Volume file: {vol}")
            new_volume_filename = f"{self.structure}_{str(i).zfill(3)}_0000.nii.gz"
            new_volume_filepath = os.path.join(self.path_to_nnunet_imagesTs, new_volume_filename)
            print(f"new volume file: {new_volume_filepath}") 

            shutil.copy(vol, new_volume_filepath)

            # Rename the testing segmentations
            print(f"segmentation file: {seg}")
            new_seg_filename = f"{self.structure}_{str(i).zfill(3)}.nii.gz"
            new_seg_filepath = os.path.join(self.path_to_nnunet_imagesTs, new_seg_filename)
            print(f"new segmentation file: {new_seg_filepath}") 

            shutil.copy(seg, new_seg_filepath)
    
    def run(self, rename_trainset=True, rename_testset=True):
        if rename_trainset:
            self.rename_train_data()
        
        if rename_testset:
            self.rename_test_data()