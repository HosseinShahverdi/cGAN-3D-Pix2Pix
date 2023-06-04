import os    
import glob
import imageio   # Convert to an image 
import shutil
import numpy as np 
import nibabel as nib
import SimpleITK as sitk  
from  PIL import Image


def nii_to_sample(file, mode, idx):

    origin = 'dataset'
    if not os.path.exists(origin):
        os.mkdir(origin)    # New Folder 

    save_to = 'dataset/extracted/'
    if not os.path.exists(save_to):
        os.mkdir(save_to)    # New Folder 
    img = nib.load(file)    # Read nii

    header = img.header
    img_fdata = img.get_fdata()

    # Contrast
    if mode == 'ct':
        img_fdata = img_fdata - np.min(img_fdata)
        img_fdata = (img_fdata/np.max(img_fdata)) * 255   
    # Start converting to an image 
    (x,y,z) = img.shape
    sumation =0
    for i in range(z):      #z Is a sequence of images 
        silce = img_fdata[:, :, i]   # You can choose which direction of slice 
        imageio.imwrite(os.path.join(save_to,'{}.png'.format(i)), silce)
        sumation += 1 
    print(sumation)
    # prepare for concantanation     
    list_files = sorted(glob.glob(os.path.abspath("dataset/extracted/*.png")),  key=len)
    index = 0
    output_dir = 'dataset/ready_oneSample/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)    # New Folder
    
    len_before =len(glob.glob(os.path.abspath("dataset/ready_oneSample/*.jpg"))) 
    for image_address in list_files:
        output_file = output_dir + str(index + len_before) + ".jpg"
        index += 1
        concat_Horizantal(image_address, image_address).save(output_file)
    shutil.rmtree(os.path.abspath('dataset/extracted'))
    return z

def extract_predict(test_images):
    # Opens a image in RGB mode
    im = Image.open(test_images)
    # Size of the image in pixels (size of original image)
    width, height = im.size
    # Setting the points for cropped image
    left = 256
    top = 0
    right = 256*2-1
    bottom = height
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
    im1 = im1.resize((167, 167))
    return im1

def concat_Horizantal(CT_img_dir, MR_img_dir):
    CT_img = Image.open(CT_img_dir)
    MR_img = Image.open(MR_img_dir)
    dst = Image.new('RGB', (CT_img.width + MR_img.width, CT_img.height))
    dst.paste(CT_img, (0, 0))
    dst.paste(MR_img, (CT_img.width, 0))
    return dst


def creat_nii(name_patient_list, z_size):
    list_files = sorted(glob.glob(os.path.abspath('pix2pix_db/test/20221203-201138/*.jpg')),  key=len)
    index = 0
    output_file_dir = 'dataset/test_predict/'
    if not os.path.exists(output_file_dir):
        os.mkdir(output_file_dir)    # New Folder 
    for image_address in list_files:       
        # for PT address
        output_file = output_file_dir + str(index) + ".jpg"
        index += 1
        extract_predict(image_address).save(output_file)
    

    # convert multiple file to nii.gz
    create_nii_file(name_patient_list, z_size)

def create_nii_file(name_patient_list,z_size):
    # convert multiple file to nii.gz
    file_names = sorted(glob.glob(os.path.abspath('dataset/test_predict/*.jpg')),  key=len)
    xlen=len(name_patient_list)
    summation = 0
    for x in range(xlen):
        slice_files =file_names[summation:(z_size[x]-1)+summation] 
        summation =+ z_size[x]  
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(slice_files)
        vol = reader.Execute()
        sitk.WriteImage(vol, os.path.abspath('nifti_res/pred_'+str(name_patient_list[x])))
    shutil.rmtree(os.path.abspath('dataset'))

def add_header(name_patient_list):
    for name_patient in name_patient_list:
        img = nib.load(os.path.abspath('pix2pix_db/nifti_header_find/'+str(name_patient)))
        gen_img = nib.load(os.path.abspath('nifti_res/pred_'+str(name_patient)))
        data   = gen_img.get_data()
        # header and affine(about spaces and image positions) from CT origin file 
        clipped_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(clipped_img, 'nifti_res/pred_'+str(name_patient))

if __name__ ==  '__main__':

    # use for test file one sample
    # First transfer nii to png
    file_name = glob.glob(os.path.abspath('Test\\oneSample\\*.gz'))
    for f in file_name:
        nii_to_sample(f, 'ct')

