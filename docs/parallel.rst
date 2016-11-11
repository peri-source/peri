***********************
Parallel image analysis
***********************

Computing on AWS
================

Barebones
---------

Starcluster
-----------

Since data analysis takes considerably longer than heuristic methods, we often
need to employ many computer resources to analyze data in a reasonable amount
of time. One such resource is provided by Amazon Webservices (AWS) Elastic
Compute cloud (EC2). This section of the documentation 

starcluster createvolume --shutdown-volume-host --name vdwdata 20 us-west-2a --bid 0.50 --image-id=ami-d732f0b7 --instance-type=m3.medium

# needs to change permissions before uploading
starcluster sshmaster cluster -u ubuntu
sudo chown ubuntu:ubuntu /data


https://calculator.s3.amazonaws.com/index.html

