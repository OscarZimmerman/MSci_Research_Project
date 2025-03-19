import numba
import swiftsimio as sw  ## swiftsimio 6.1.1
import numpy as np
import unyt ## unyt 2.9.5
import h5py
import woma    ## woma 1.2.0 

R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg m^-3 
G = 6.67408e-11  # m^3 kg^-1 s^-2



def load_to_woma(snapshot):
    # Load
    data    = sw.load(snapshot)
    
    data.gas.coordinates.convert_to_mks()
    data.gas.velocities.convert_to_mks()
    data.gas.smoothing_lengths.convert_to_mks()
    data.gas.masses.convert_to_mks()
    data.gas.densities.convert_to_mks()
    data.gas.pressures.convert_to_mks()
    data.gas.internal_energies.convert_to_mks()
    box_mid = 0.5*data.metadata.boxsize[0].to(unyt.m)

    pos     = np.array(data.gas.coordinates - box_mid)
    vel     = np.array(data.gas.velocities)
    h       = np.array(data.gas.smoothing_lengths)
    m       = np.array(data.gas.masses)
    rho     = np.array(data.gas.densities)
    p       = np.array(data.gas.pressures)
    u       = np.array(data.gas.internal_energies)
    matid   = np.array(data.gas.material_ids)
    
    pos_centerM = np.sum(pos * m[:,np.newaxis], axis=0) / np.sum(m)
    vel_centerM = np.sum(vel * m[:,np.newaxis], axis=0) / np.sum(m)
    
    pos -= pos_centerM
    vel -= vel_centerM
    
    xy = np.hypot(pos[:,0],pos[:,1])
    r  = np.hypot(xy,pos[:,2])
    r  = np.sort(r)
    R  = np.mean(r[-100:])
    
    return pos, vel, h, m, rho, p, u, matid, R


from sklearn.cluster import DBSCAN
import numpy as np

def load_to_woma_com(snapshot):
    # Load data
    data = sw.load(snapshot)
    
    data.gas.coordinates.convert_to_mks()
    data.gas.velocities.convert_to_mks()
    data.gas.smoothing_lengths.convert_to_mks()
    data.gas.masses.convert_to_mks()
    data.gas.densities.convert_to_mks()
    data.gas.pressures.convert_to_mks()
    data.gas.internal_energies.convert_to_mks()
    
    box_mid = 0.5 * data.metadata.boxsize[0].to(unyt.m)
    
    # Convert positions and velocities
    pos = np.array(data.gas.coordinates - box_mid)
    vel = np.array(data.gas.velocities)
    h = np.array(data.gas.smoothing_lengths)
    m = np.array(data.gas.masses)
    rho = np.array(data.gas.densities)
    p = np.array(data.gas.pressures)
    u = np.array(data.gas.internal_energies)
    matid = np.array(data.gas.material_ids)
    
    # DBSCAN clustering to find groups of particles (use distance threshold and minimum particles per cluster)
    db = DBSCAN(eps=1000, min_samples=10) 
    labels = db.fit_predict(pos)
    
    # Find the largest cluster by mass
    unique_labels = set(labels)
    largest_cluster_label = max(unique_labels, key=lambda label: np.sum(m[labels == label]) if label != -1 else 0)
    
    # Get particles in the largest cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    
    # Filter out particles from the largest cluster
    pos = pos[largest_cluster_indices]
    vel = vel[largest_cluster_indices]
    h = h[largest_cluster_indices]
    m = m[largest_cluster_indices]
    rho = rho[largest_cluster_indices]
    p = p[largest_cluster_indices]
    u = u[largest_cluster_indices]
    matid = matid[largest_cluster_indices]
    
    com = np.sum(pos * m[:, np.newaxis], axis=0) / np.sum(m)

    
    # Calculate the radius of the largest body (mean distance of particles from the COM)
    distances = np.linalg.norm(pos - com, axis=1)
    R = np.mean(distances)  
    
    M = np.sum(m)
    
    return pos, vel, h, m, rho, p, u, matid, R, com, M


def filter_particles_by_com_distance(input_file, output_file, distance_threshold):
    with h5py.File(input_file, "r") as file:
        gas_particles_group = file["GasParticles"]
        
        # Read the coordinates and masses datasets
        coordinates = gas_particles_group["Coordinates"][:]
        masses = gas_particles_group["Masses"][:]
        
        # Calculate the center of mass
        center_of_mass = np.sum(coordinates * masses[:, np.newaxis], axis=0) / np.sum(masses)
        print(f"Center of Mass: {center_of_mass}")
        
        # Compute distances from the center of mass
        distances = np.linalg.norm(coordinates - center_of_mass, axis=1)
        
        # Create a mask for particles within the threshold distance
        within_threshold = distances <= distance_threshold
        num_particles_removed = np.sum(~within_threshold)
        print(f"Number of particles removed: {num_particles_removed}")
        
        # Ensure the mask is a boolean array
        within_threshold = within_threshold.astype(bool)

        # Copy the filtered data to a new HDF5 file
        with h5py.File(output_file, "w") as output:
            # Replicate the structure of the input file
            def copy_and_filter_group(src_group, dst_group, within_threshold):
                for name, item in src_group.items():
                    if isinstance(item, h5py.Group):
                        new_group = dst_group.create_group(name)
                        for attr_name, attr_value in item.attrs.items():
                            new_group.attrs[attr_name] = attr_value
                        copy_and_filter_group(item, new_group, within_threshold)  # Pass within_threshold
                    elif isinstance(item, h5py.Dataset):
                        if len(item.shape) > 0 and item.shape[0] == len(within_threshold):
                            
                            # Ensure within_threshold is a boolean array
                            filtered_data = np.array(item)[within_threshold.astype(bool)]
                            dst_group.create_dataset(name, data=filtered_data)
                        else:
                            dst_group.create_dataset(name, data=item[:])  # Copy datasets without filtering
                        # Copy attributes
                        for attr_name, attr_value in item.attrs.items():
                            dst_group[name].attrs[attr_name] = attr_value

            
            copy_and_filter_group(file, output, within_threshold)

gamma = 0.11
b = 0.55

filter_particles_by_com_distance("b=0.65/n=4/coll1_0225.hdf5", "rel4-1.hdf5", 50)

loc_tar = 'rel4-1.hdf5'
loc_imp = 'impactor4_0000.hdf5'

#reset variables
pos_tar = 0
vel_tar = 0
h_tar = 0
m_tar = 0
rho_tar = 0
p_tar = 0
u_tar = 0
matid_tar = 0
R_tar = 0

pos_imp = 0
vel_imp = 0
h_imp = 0
m_imp = 0
rho_imp = 0
p_imp = 0
u_imp = 0
matid_imp = 0
R_imp = 0

pos_tar_com = 0
vel_tar_com = 0
h_tar_com = 0 
m_tar_com = 0
rho_tar_com = 0 
p_tar_com = 0
u_tar_com = 0 
matid_tar_com = 0 
R_tar_com = 0 
com = 0 
mass = 0

pos_tar, vel_tar, h_tar, m_tar, rho_tar, p_tar, u_tar, matid_tar, R_tar = load_to_woma(loc_tar)
pos_imp, vel_imp, h_imp, m_imp, rho_imp, p_imp, u_imp, matid_imp, R_imp = load_to_woma(loc_imp)
pos_tar_com, vel_tar_com, h_tar_com, m_tar_com, rho_tar_com, p_tar_com, u_tar_com, matid_tar_com, R_tar_com, com, mass = load_to_woma_com(loc_tar) # Load in variables for only the largest body in the target file

M_t = np.sum(m_tar_com)
M_i = np.sum(m_imp)
R_t = R_tar_com
R_i = R_imp

# Mutual escape speed
v_esc = np.sqrt(2 * G * (M_t + M_i) / (R_t + R_i))

# Initial position and velocity of the target
A1_pos_t = np.array([0., 0., 0.])
A1_vel_t = np.array([0., 0., 0.])

A1_pos_i, A1_vel_i = woma.impact_pos_vel_b_v_c_t(
    b       = b,
    v_c     = 1.1*v_esc, 
    t       = 3600, 
    R_t     = R_t, 
    R_i     = R_i, 
    M_t     = M_t, 
    M_i     = M_i,
)

A1_pos_com = (M_t * A1_pos_t + M_i * A1_pos_i) / (M_t + M_i)
A1_pos_t -= A1_pos_com
A1_pos_i -= A1_pos_com

# Centre of momentum
A1_vel_com = (M_t * A1_vel_t + M_i * A1_vel_i) / (M_t + M_i)
A1_vel_t -= A1_vel_com
A1_vel_i -= A1_vel_com

pos_tar += A1_pos_t
vel_tar[:] += A1_vel_t

pos_imp += A1_pos_i
vel_imp[:] += A1_vel_i


with h5py.File(f"n=4_coll2.hdf5", "w") as f:
    woma.save_particle_data(
        f,
        np.append(pos_tar, pos_imp, axis=0),
        np.append(vel_tar, vel_imp, axis=0),
        np.append(m_tar, m_imp),
        np.append(h_tar, h_imp),
        np.append(rho_tar, rho_imp),
        np.append(p_tar, p_imp),
        np.append(u_tar, u_imp),
        np.append(matid_tar, matid_imp),
        boxsize=500 * R_earth, 
        file_to_SI=woma.Conversions(M_earth, R_earth, 1),
        
    )
