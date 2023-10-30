provider "aws" {
  region  = "ap-southeast-2"
  profile = var.aws_profile
}

resource "aws_key_pair" "dev_key" {
  key_name   = "dev_key"
  public_key = file("../secrets/dev_key.pub")
}

resource "aws_security_group" "dev_sg" {
  name        = "dev_sg"
  description = "Allow SSH and HTTP"
  vpc_id      = aws_vpc.dev_server.id

  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }

  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }
}

resource "aws_instance" "dev_server" {
  ami           = "ami-09b402d0a0d6b112b"
  instance_type = "t2.xlarge"
  tags = {
    Name = "Remote Dev Server"
  }

  root_block_device {
    volume_size = 30
    volume_type = "gp2"
  }

  key_name               = aws_key_pair.dev_key.key_name
  vpc_security_group_ids = [aws_security_group.dev_sg.id]
  subnet_id              = aws_subnet.dev_server.id
  user_data              = file("scripts/startup.sh")
}
