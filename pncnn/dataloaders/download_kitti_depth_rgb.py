# code-checked
# server-checked

import os
import requests
import improved_normal_inference.config as config
download_rgb = False

# Where you want to download kitti-depth
kitti_depth_path = config.synthetic_data / "kitti_depth"

# Where you want to download the RGB images for kitti-depth
kitti_rgb_path = config.synthetic_data / "kitti_rgb"

############## Download KITTI-Depth ########################
print("======> Downloading KITTI-Depth <======")
if not os.path.exists(kitti_depth_path):
    os.makedirs(kitti_depth_path)

# download the zip files:

import requests, zipfile, io
r = requests.get("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(kitti_depth_path)


# download_url("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip",
#              kitti_depth_path)
# download_url("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip",
#              kitti_depth_path)

# unzip:
os.system("unzip %s/data_depth_annotated.zip -d %s" % (kitti_depth_path, kitti_depth_path))
os.system("unzip %s/data_depth_velodyne.zip -d %s" % (kitti_depth_path, kitti_depth_path))
os.system("unzip %s/data_depth_selection.zip -d %s" % (kitti_depth_path, kitti_depth_path))

if download_rgb:
    print("======> Downloading KITTI-RGB <======")

    os.system("mkdir %s" % kitti_rgb_path)

    train_dirs = os.listdir(kitti_depth_path + "/train")  # (contains "2011_09_26_drive_0001_sync" and so on)
    val_dirs = os.listdir(kitti_depth_path + "/selval")

    # Create "train" and "selval" dir for RGB
    rgb_train_dir = os.path.join(kitti_rgb_path, "train")
    rgb_val_dir = os.path.join(kitti_rgb_path, "selval")
    os.system("mkdir %s" % rgb_train_dir)
    os.system("mkdir %s" % rgb_val_dir)

    print("num train dirs: %d" % len(train_dirs))
    print("num selval dirs: %d" % len(val_dirs))

    # Training set
    for step, dir_name in enumerate(train_dirs):
        print("########################### Training set #########################################")
        print("step %d/%d" % (step + 1, len(train_dirs)))
        print(dir_name)
        dir_name_no_sync = dir_name.split("_sync")[0]  # (dir_name_no_sync == "2011_09_26_drive_0001")

        # download the zip file:
        os.system("wget -P %s https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/%s/%s.zip" % (
        kitti_rgb_path, dir_name_no_sync, dir_name))

        # unzip:
        os.system("unzip %s.zip -d %s" % (os.path.join(kitti_rgb_path, dir_name), kitti_rgb_path))

        # move to rgb dir
        zip_dir = dir_name.split('_drive')[0]
        os.system("mv %s %s" % (os.path.join(kitti_rgb_path, zip_dir, dir_name), rgb_train_dir))
        os.system("rm -rf %s" % os.path.join(kitti_rgb_path, zip_dir))

    # Validation set
    for step, dir_name in enumerate(val_dirs):
        print("########################### Validation set #########################################")
        print("step %d/%d" % (step + 1, len(val_dirs)))
        print(dir_name)
        dir_name_no_sync = dir_name.split("_sync")[0]  # (dir_name_no_sync == "2011_09_26_drive_0001")

        # download the zip file:
        os.system("wget -P %s https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/%s/%s.zip" % (
        kitti_rgb_path, dir_name_no_sync, dir_name))

        # unzip:
        os.system("unzip %s.zip -d %s" % (os.path.join(kitti_rgb_path, dir_name), kitti_rgb_path))

        # move to rgb dir
        # move to rgb dir
        zip_dir = dir_name.split('_drive')[0]
        os.system("mv %s %s" % (os.path.join(kitti_rgb_path, zip_dir, dir_name), rgb_val_dir))
        os.system("rm -rf %s" % os.path.join(kitti_rgb_path, zip_dir))
