import os
import subprocess
import os.path as osp
from zipfile import ZipFile

import hiyapyco as hyaml
import omg
import shutil


### Load Config ###
def load_config(filenames):
    # list all config files
    file_paths = ["resources/map_configs/{0}.yaml".format(file) for file in filenames]

    # Load all yaml files listed in flnms and combine into a single dictionary, recursively combining keys
    cfg = hyaml.load(file_paths,method=hyaml.METHOD_MERGE)
    return cfg
            

### Building ACS files ###
def make_acs(cfg,actor_names,object_types,actor_num_textures):
    object_variables_acs = ""

    object_variables_template = """
    // {type} variables
    {type}_unique = {unique};
    {type}_init = {init};
    {type}_delay = {delay};
    """

    for type in object_types:
        type_cfg = cfg['objects'][type]
        object_variables_acs += object_variables_template.format(type=type,
                                                                 unique=len(type_cfg['actors']),
                                                                 init=type_cfg['init'],
                                                                 delay=type_cfg['delay'])


    actor_arrays_initialization = ""
    actor_arrays_template ="""
    actor_names[{index}] = "{actor_name}";
    actor_num_textures[{index}] = {num_textures};
    """
    for i,(actor_name,num_textures) in enumerate(zip(actor_names,actor_num_textures)):
        actor_arrays_initialization += actor_arrays_template.format(index=i,
                                                                    actor_name=actor_name,
                                                                    num_textures=num_textures)

    with open("resources/templates/acs.txt") as f:
        acs = f.read().format(metabolic_delay=cfg['metabolic']['delay'], metabolic_damage=cfg['metabolic']['damage'], object_variables=object_variables_acs, array_variables = actor_arrays_initialization)

    return acs

### Building Decorate Files ###

def actor_code(i,j):
    # Convert k to caps alpha string but skip 'F'
    if i >= 5:
        return chr(65 + 1 + i) + texture_code(j)
    else:
        return chr(65 + i) + texture_code(j)

def texture_code(j):
    # Convert k to a 3-digit alpha all-caps string
    return chr(65 + j // 26 ** 2) + chr(65 + (j // 26) % 26) + chr(65 + j % 26)

def make_decorate_include(actor_names):
    decorate = ""
    for actor_name in actor_names:
        decorate += "#include \"actors/{0}.dec\"\n".format(actor_name)
    return decorate

def make_decorate(cfg,templates,actor_name,typ, sprite_names):

    actor_cfg = cfg['actors'][actor_name]
    state_template="Texture{index}: {texture_code} A -1\n\t"
    states=""

    for i, sprite_name in enumerate(sprite_names):
        states += state_template.format(index=i, texture_code=sprite_name)

    if typ == "nourishment":
        decorate=templates[typ].format(name=actor_name,healing=actor_cfg['healing'],states_definitions=states)
    elif typ == "poison":
        decorate=templates[typ].format(name=actor_name,damage=actor_cfg['damage'],states_definitions=states)
    elif typ == "obstacle":
        decorate=templates[typ].format(name=actor_name, radius=24,states_definitions=states)
    elif typ == "distractor":
        decorate=templates[typ].format(name=actor_name,states_definitions=states)
    else:
        raise ValueError("Invalid actor type: {0}".format(typ))

    return decorate

def load_decorate_templates():
    templates={}

    with open("resources/templates/nourishment-dec.txt") as f:
        templates["nourishment"] = f.read()
    with open("resources/templates/poison-dec.txt") as f:
        templates["poison"] = f.read()
    with open("resources/templates/obstacle-dec.txt") as f:
        templates["obstacle"] = f.read()
    with open("resources/templates/distractor-dec.txt") as f:
        templates["distractor"] = f.read()
    return templates

def get_pngs(base_path, png_pths):
    pngs = []
    for png_path in png_pths:
        full_path = osp.join(base_path,png_path)
        # if pngpth is a png, add it
        if png_path.endswith(".png"):
            pngs.append(full_path)
        elif osp.isdir(full_path):
            for root, _, files in os.walk(full_path):
                for file in files:
                    if file.endswith(".png"):
                        pngs.append(osp.join(root,file))
    return pngs

### Creating Scenarios ###
def make_scenario(config_files, scenario_name=None):

    # Preloading
    cfg = load_config(config_files)
    object_types = cfg['objects'].keys()

    if scenario_name is None:
        scenario_name = "-".join(config_files)

    # Base directories
    resource_dir = "resources"
    scenario_dir = "scenarios"

    # Inupt Directories & Files
    input_base_dir = osp.join(resource_dir,"base")

    # Create Zip for output
    out_file = osp.join(scenario_dir,scenario_name) + ".zip"
    if osp.exists(out_file):
        os.remove(out_file)
    scenario_zip = ZipFile(out_file, "x")

    # Create directories in zip
    scenario_zip.mkdir("acs")
    scenario_zip.mkdir("maps")
    scenario_zip.mkdir("sprites")
    scenario_zip.mkdir("actors")
    scenario_zip.mkdir("textures")

    # Textures
    scenario_zip.write(osp.join(input_base_dir,"grass.png"),osp.join("textures","GRASS.png"))
    scenario_zip.write(osp.join(input_base_dir,"wind.png"), osp.join("textures","WIND.png"))

    # Copy Data to Root
    scenario_zip.write(osp.join(input_base_dir,"MAPINFO.txt"), "MAPINFO.txt")

    # Building decorate and loading textures
    dec_templates=load_decorate_templates()
    actor_names = []
    actor_num_textures = []

    actor_idx = 0
    for type in object_types:
        type_cfg = cfg['objects'][type]
        for actor_name in type_cfg['actors']:

            # get all pngs listend in pngpths and subdirs
            png_paths = type_cfg['actors'][actor_name]['textures']
            pngs = get_pngs(osp.join(resource_dir,"textures"), png_paths)

            num_textures = len(pngs)

            sprite_names = [actor_code(actor_idx,i) for i in range(num_textures)]
            # Add pngs as sprites
            [scenario_zip.write(png,osp.join("sprites",sprite_names[j] + "A0.png")) for j, png in enumerate(pngs)]

            decorate = make_decorate(type_cfg,dec_templates, actor_name,type,sprite_names)

            scenario_zip.writestr(osp.join("actors", actor_name+".dec"), decorate)

            actor_idx += 1
            actor_names.append(actor_name)
            actor_num_textures.append(num_textures)

    # Write decorate include to root
    decorate = make_decorate_include(actor_names)
    scenario_zip.writestr("DECORATE.txt", decorate)

    ## ACS ##

    # Defining paths
    build_path = osp.join(scenario_dir, "build")
    if osp.exists(build_path):
        shutil.rmtree(build_path)
    os.mkdir(build_path)

    retinal_acs_path = osp.join(input_base_dir,"acs","retinal.acs")
    map_acs_path = osp.join(build_path,scenario_name)+".acs"
    retinal_compiled_path = osp.join(build_path, "retinal.o")
    map_compiled_path =map_acs_path[:-3]+"o" # Replace ".acs" ending with ".o"

    # Write ACS
    with open(map_acs_path,'w') as f:
        acs = make_acs(cfg,actor_names,object_types,actor_num_textures)
        f.write(acs)

    # Compile ACS
    subprocess.call(["acc", "-i","/usr/share/acc", retinal_acs_path, retinal_compiled_path])
    subprocess.call(["acc", "-i","/usr/share/acc", "-i", input_base_dir, map_acs_path])

    # For completeness, add retinal and map acs to zip
    scenario_zip.write(retinal_compiled_path,osp.join("acs","retinal.o"))
    scenario_zip.write(retinal_acs_path,osp.join("acs","retinal.acs"))
    scenario_zip.write(map_acs_path, "behavior.acs")

    # Map Wad
    wad = omg.WAD()
    map_lump = omg.LumpGroup()
    map_lump["TEXTMAP"] = omg.Lump(from_file=osp.join(input_base_dir,"TEXTMAP.txt"))
    map_lump["BEHAVIOR"] = omg.Lump(from_file=map_compiled_path)
    wad.udmfmaps["MAP01"] = omg.UMapEditor(map_lump).to_lumps()

    # Save wad to map and add to zip
    map_path = osp.join(build_path,"MAP01.wad")
    wad.to_file(map_path)
    scenario_zip.write(map_path, osp.join("maps", "MAP01.wad"))

    # Cleanup
    shutil.rmtree(build_path)

    # Copy vizdoom config
    config_name = scenario_name + ".cfg"
    shutil.copy(osp.join(input_base_dir,"vizdoom.cfg"),osp.join(scenario_dir,config_name))
    # add doom_scenario_path to beginning of cfg
    with open(osp.join(scenario_dir,config_name),'r') as f:
        cfgtxt = f.read()
    cfgtxt = """# Settings copied from resources/base/vizdoom.cfg
doom_scenario_path = {scenario_name}.zip

""".format(scenario_name=scenario_name) + cfgtxt
    with open(osp.join(scenario_dir,config_name),'w') as f:
        f.write(cfgtxt)
