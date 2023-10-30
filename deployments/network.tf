resource "aws_vpc" "dev_server" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = {
    Name = "dev-server-vpc"
  }
}

resource "aws_subnet" "dev_server" {
  vpc_id            = aws_vpc.dev_server.id
  cidr_block        = cidrsubnet(aws_vpc.dev_server.cidr_block, 3, 1)
  availability_zone = "ap-southeast-2a"
  tags = {
    Name = "dev-server-subnet"
  }
}

resource "aws_eip" "dev-server" {
  instance = aws_instance.dev_server.id
}

resource "aws_internet_gateway" "dev_server" {
  vpc_id = aws_vpc.dev_server.id
  tags = {
    Name = "dev-server-igw"
  }
}

resource "aws_route_table" "dev_server" {
  vpc_id = aws_vpc.dev_server.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.dev_server.id
  }
  tags = {
    Name = "dev-server-rt"
  }
}

resource "aws_route_table_association" "dev_server" {
  subnet_id      = aws_subnet.dev_server.id
  route_table_id = aws_route_table.dev_server.id
}