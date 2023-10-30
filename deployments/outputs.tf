output "instance_ip" {
  value       = aws_eip.dev-server.public_ip
  description = "The public IP of the dev server"
}
