"""slc.py: Functions for SLC related operations."""


# def intf_to_slc(mother_slc, ifg, blockwise=False):
#     return (ifg * mother_slc.conj())/(np.abs(mother_slc)**2)

# def intf_to_slc_stack(mother_slc, ifgs, blockwise=False):
#     len(ifgs.time)

#     slc_prime_all = []
#     for ifg in ifgs.time:
#         slc_prime_all.append(intf_to_slc(mother_slc, ifgs[:,:,ifg]))
#     return slc_prime_all
