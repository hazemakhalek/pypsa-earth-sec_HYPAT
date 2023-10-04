
import ruamel.yaml

file_paths = ['/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/config_2030_cons.yaml',
            '/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/config_2030_opt.yaml',
            '/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/config_2030_real.yaml',
            '/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/config_2050_cons.yaml',
            '/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/config_2050_opt.yaml',
            '/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/config_2050_real.yaml']


# Read the YAML file
yaml = ruamel.yaml.YAML()

for file_path in file_paths:
    with open(file_path, 'r') as file:
        yaml_content = yaml.load(file)

    # Modify the desired line (e.g., change a value)

    yaml_content['export']['h2export'] = yaml_content['export']['h2export_all_quantities'][2:]

    # Write the updated YAML back to the file
    with open(file_path, 'w') as file:
        yaml.dump(yaml_content, file)