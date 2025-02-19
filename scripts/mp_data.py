from pymatgen.ext.matproj import MPRester

with MPRester("ZQQUqL8DRN3tMv5VyrzrA3bwZv623rji") as mpr:
    structure = mpr.get_structure_by_material_id("mp-5229")

structure.to(filename="mp-5229.cif")
