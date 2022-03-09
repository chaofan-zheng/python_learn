# Mysql
https://blog.csdn.net/skh2015java/article/details/80156278
## 安装
- sudo apt-get install mysql-server
- apt-get install mysql-client
- sudo apt-get install libmysqlclient-dev
## 检查状态
- sudo netstat -tap | grep mysql
- sudo service mysql status
## 启动
- sudo service mysql start/stop/restart
- sudo mysqladmin -u root -p password
- mysql -h 主机地址 -u 用户名 -p 
  