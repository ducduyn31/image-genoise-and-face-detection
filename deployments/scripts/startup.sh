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
sudo bash ~/ec2-user/anaconda.sh -b -p /home/anaconda3
rm ~/ec2-user/anaconda.sh
echo "export PATH=/home/anaconda3/bin:$PATH" | sudo tee -a /etc/profile.d/anaconda.sh
echo "source /home/anaconda3/bin/activate" | sudo tee -a /etc/profile.d/anaconda.sh

# Setup Jupyter notebook
mkdir -p /home/ec2-user/projects

echo "[Unit]
Description=Jupyter Notebook Server

[Service]
User=ec2-user
Group=ec2-user
ExecStart=/home/anaconda3/bin/jupyter-notebook --ip=0.0.0.0 --no-browser
WorkingDirectory=/home/ec2-user/projects
Restart=always

[Install]
WantedBy=multi-user.target" | sudo tee /etc/systemd/system/jupyter.service

# Enable and start the Jupyter service
sudo systemctl enable jupyter.service
sudo systemctl start jupyter.service
