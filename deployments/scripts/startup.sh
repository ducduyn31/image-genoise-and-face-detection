#!/bin/bash

## Create /home/ec2-user/projects if not exists
#mkdir -p /home/ec2-user/projects
#
## Get the UUID of the EBS volume at /dev/sdf
#UUID=$(sudo blkid -s UUID -o value /dev/sdf)
#
## Create a new fstab to mount to /home
#FSTAB_ENTRY="UUID=$UUID /home/ec2-user/projects ext4 defaults,nofail 0 2"
#
## Append the fstab entry to the fstab file
#echo $FSTAB_ENTRY | sudo tee -a /etc/fstab
#
## Mount the new volume
#sudo mount -a

# Download latest anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -O /home/ec2-user/anaconda.sh
bash ~/ec2-user/anaconda.sh -b -p /home/anaconda3
rm ~/ec2-user/anaconda.sh
echo "export PATH=/home/anaconda3/bin:$PATH" | sudo tee -a /etc/profile.d/anaconda.sh
echo "source /home/anaconda3/bin/activate" | sudo tee -a /etc/profile.d/anaconda.sh
