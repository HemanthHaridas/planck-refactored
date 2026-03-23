import basis_set_exchange as bse

# Get a list of all basis sets
basis_list  = bse.get_all_basis_names()

# Get Pople-style basis sets
pople_list  = [name for name in basis_list if "G" in name or "cc" in name]

for name in pople_list:
    print(name)
