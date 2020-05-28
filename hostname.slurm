#!/bin/bash
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 1-00:00 # Runtime in days-hours:minutes
#SBATCH -p shared # Partition to submit to
#SBATCH --mem=100 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append # Append when writing files
#SBATCH -o %j.out # Standard out goes to this file
#SBATCH -e %j.err # Standard err goes to this filehostname
module load Anaconda3/2019.10
hostid=$(grbprobe | grep "HOSTID")
hostname=$(grbprobe | grep "HOSTNAME")
rm /n/home09/anyazhang/gurobi902/gurobi.lic
curl --data "id=d1204ecc-a05c-11ea-8cca-0a7c4f30bdbe
&hostname=$hostname&hostid=$hostid&username=anyazhang&os=linux64&localdate=2020-05-27&version=9&sockets=2" -X POST http://apps.gurobi.com/keyserver > /n/home09/anyazhang/gurobi902/gurobi.lic
sed -i '1,2d' /n/home09/anyazhang/gurobi902/gurobi.lic
python3 run_pov_utility.py