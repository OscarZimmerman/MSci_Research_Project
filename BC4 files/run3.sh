#!/bin/bash -l

# Get the initial conditions if they are not present.
##if [ ! -target.hdf5 ]
##then
##    echo "Fetching initial conditions file for the Earth impact example..."
##    ./get_init_cond.sh
##fi



#SBATCH -J swift_impact_1node28cpu_n=3_b=0.55
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=ds21754@bristol.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -t 7-00:00:00


##SBATCH --exclusive


module load openmpi
module load hdf5
module load languages/python/3.12.3


source myenv/bin/activate

# Combine planets

python combine_planets3.py # First collision

# Run SWIFT
../../../swift -a -s -G -t 14 earth_impact3-1.yml 2>&1 | tee output.log

python combine_planets3-1.py # Second collision

../../../swift -a -s -G -t 14 earth_impact3-2.yml 2>&1 | tee output.log

python combine_planets3-2.py # Third collision

../../../swift -a -s -G -t 14 earth_impact3-3.yml 2>&1 | tee output.log
