import os
import os.path as osp
import shutil
import subprocess
from zipfile import ZipFile

import hiyapyco as hyaml
import omg


### Load Config ###
def load_config(filenames):
    # list all config files
    file_pths = ["resources/map_configs/{0}.yaml".format(file) for file in filenames]

    # Load all yaml files listed in flnms and combine into a single dictionary, recursively combining keys
    cfg = hyaml.load(file_pths, method=hyaml.METHOD_MERGE)
    return cfg


### Building ACS files ###
def make_acs(cfg, object_types, actor_names, num_textures):
    object_variables_acs = ""

    object_variables_template = """
    // {type} variables
    {type}_unique = {unique};
    {type}_init = {init};
    {type}_delay = {delay};
    """

    for type in object_types:
        type_cfg = cfg["objects"][type]
        object_variables_acs += object_variables_template.format(
            type=type,
            unique=len(type_cfg["actors"]),
            init=type_cfg["init"],
            delay=type_cfg["delay"],
        )

    actor_arrays_initialization = ""
    actor_arrays_template = """
    actor_names[{index}] = "{actor_name}";
    actor_num_textures[{index}] = {num_textures};
    """
    for i, (actor_name, num_textures) in enumerate(zip(actor_names, num_textures)):
        actor_arrays_initialization += actor_arrays_template.format(
            index=i, actor_name=actor_name, num_textures=num_textures
        )

    with open("resources/templates/acs.txt") as f:
        acs = f.read().format(
            metabolic_delay=cfg["metabolic"]["delay"],
            metabolic_damage=cfg["metabolic"]["damage"],
            object_variables=object_variables_acs,
            array_variables=actor_arrays_initialization,
        )

    return acs


### Building Decorate Files ###
def actor_code(i, j):
    # Convert k to caps alpha string but skip 'F'
    if i >= 5:
        return chr(65 + 1 + i) + texture_code(j)
    else:
        return chr(65 + i) + texture_code(j)


def texture_code(j):
    # Convert k to a 3-digit alpha all-caps string
    return chr(65 + j // 26**2) + chr(65 + (j // 26) % 26) + chr(65 + j % 26)


def make_decorate_include(actor_names):
    decorate = ""
    for actor_name in actor_names:
        decorate += '#include "actors/{0}.dec"\n'.format(actor_name)
    return decorate


def make_decorate(cfg, templates, actor_name, typ, sprite_names):
    actor_cfg = cfg["actors"][actor_name]
    state_template = "Texture{index}: {texture_code} A -1\n\t"
    states = ""

    for i, sprite_name in enumerate(sprite_names):
        states += state_template.format(index=i, texture_code=sprite_name)

    if typ == "nourishment":
        decorate = templates[typ].format(
            name=actor_name, healing=actor_cfg["healing"], states_definitions=states
        )
    elif typ == "poison":
        decorate = templates[typ].format(
            name=actor_name, damage=actor_cfg["damage"], states_definitions=states
        )
    elif typ == "obstacle":
        decorate = templates[typ].format(
            name=actor_name, radius=24, states_definitions=states
        )
    elif typ == "distractor":
        decorate = templates[typ].format(name=actor_name, states_definitions=states)
    else:
        raise ValueError("Invalid actor type: {0}".format(typ))

    return decorate


def load_decorate_templates():
    templates = {}

    with open("resources/templates/nourishment-dec.txt") as f:
        templates["nourishment"] = f.read()
    with open("resources/templates/poison-dec.txt") as f:
        templates["poison"] = f.read()
    with open("resources/templates/obstacle-dec.txt") as f:
        templates["obstacle"] = f.read()
    with open("resources/templates/distractor-dec.txt") as f:
        templates["distractor"] = f.read()
    return templates


def get_pngs(base_pth, png_pths):
    pngs = []
    for png_pth in png_pths:
        full_pth = osp.join(base_pth, png_pth)
        # if pngpth is a png, add it
        if png_pth.endswith(".png"):
            pngs.append(full_pth)
        elif osp.isdir(full_pth):
            for root, _, files in os.walk(full_pth):
                for file in files:
                    if file.endswith(".png"):
                        pngs.append(osp.join(root, file))
    return pngs


### Creating Scenarios ###
def make_scenario(config_files, scenario_name=None):
    # Preloading
    cfg = load_config(config_files)
    object_types = cfg["objects"].keys()

    if scenario_name is None:
        scenario_name = "-".join(config_files)

    # Base directories
    resource_dir = "resources"
    scenario_dir = "scenarios"

    # Inupt Directories & Files
    base_dir = osp.join(resource_dir, "base")

    # Create Zip for output
    out_file = osp.join(scenario_dir, scenario_name) + ".zip"
    if osp.exists(out_file):
        os.remove(out_file)
    s_zip = ZipFile(out_file, "x")

    # Create directories in zip
    s_zip.mkdir("acs")
    s_zip.mkdir("maps")
    s_zip.mkdir("sprites")
    s_zip.mkdir("actors")
    s_zip.mkdir("textures")

    # Textures
    s_zip.write(osp.join(base_dir, "grass.png"), osp.join("textures", "GRASS.png"))
    s_zip.write(osp.join(base_dir, "wind.png"), osp.join("textures", "WIND.png"))

    # Copy Data to Root
    s_zip.write(osp.join(base_dir, "MAPINFO.txt"), "MAPINFO.txt")

    # Building decorate and loading textures
    dec_templates = load_decorate_templates()
    actor_names = []
    actor_num_textures = []

    actor_idx = 0
    for type in object_types:
        type_cfg = cfg["objects"][type]
        for actor_name in type_cfg["actors"]:
            # get all pngs listend in pngpths and subdirs
            png_pths = type_cfg["actors"][actor_name]["textures"]
            pngs = get_pngs(osp.join(resource_dir, "textures"), png_pths)

            num_textures = len(pngs)

            sprite_names = [actor_code(actor_idx, i) for i in range(num_textures)]
            # Add pngs as sprites
            for j, png in enumerate(pngs):
                s_zip.write(png, osp.join("sprites", sprite_names[j] + "A0.png"))

            dec = make_decorate(type_cfg, dec_templates, actor_name, type, sprite_names)
            s_zip.writestr(osp.join("actors", actor_name + ".dec"), dec)

            actor_idx += 1
            actor_names.append(actor_name)
            actor_num_textures.append(num_textures)

    # Write decorate include to root
    decorate = make_decorate_include(actor_names)
    s_zip.writestr("DECORATE.txt", decorate)

    ## ACS ##

    # Defining pths
    build_pth = osp.join(scenario_dir, "build")
    if osp.exists(build_pth):
        shutil.rmtree(build_pth)
    os.mkdir(build_pth)

    retinal_acs_pth = osp.join(base_dir, "acs", "retinal.acs")
    map_acs_pth = osp.join(build_pth, scenario_name) + ".acs"
    retinal_comp_pth = osp.join(build_pth, "retinal.o")
    map_comp_pth = map_acs_pth[:-3] + "o"  # Replace ".acs" ending with ".o"

    # Write ACS
    acs = make_acs(cfg, object_types, actor_names, actor_num_textures)
    with open(map_acs_pth, "w") as f:
        f.write(acs)

    # Compile ACS
    subprocess.call(["acc", "-i", "/usr/share/acc", retinal_acs_pth, retinal_comp_pth])
    subprocess.call(["acc", "-i", "/usr/share/acc", "-i", base_dir, map_acs_pth])

    # For completeness, add retinal and map acs to zip
    s_zip.write(retinal_comp_pth, osp.join("acs", "retinal.o"))
    s_zip.write(retinal_acs_pth, osp.join("acs", "retinal.acs"))
    s_zip.write(map_acs_pth, "behavior.acs")

    # Map Wad
    wad = omg.WAD()
    map_lump = omg.LumpGroup()
    map_lump["TEXTMAP"] = omg.Lump(from_file=osp.join(base_dir, "TEXTMAP.txt"))
    map_lump["BEHAVIOR"] = omg.Lump(from_file=map_comp_pth)
    wad.udmfmaps["MAP01"] = omg.UMapEditor(map_lump).to_lumps()

    # Save wad to map and add to zip
    map_pth = osp.join(build_pth, "MAP01.wad")
    wad.to_file(map_pth)
    s_zip.write(map_pth, osp.join("maps", "MAP01.wad"))

    # Cleanup
    shutil.rmtree(build_pth)

    # Copy vizdoom config
    config_name = scenario_name + ".cfg"
    # add doom_scenario_pth to beginning of cfg
    cfg_template = """# Settings copied from resources/base/vizdoom.cfg
doom_scenario_path = {scenario_name}.zip

"""
    with open(osp.join(base_dir, "vizdoom.cfg"), "r") as f:
        cfgtxt = cfg_template.format(scenario_name=scenario_name) + f.read()
    with open(osp.join(scenario_dir, config_name), "w") as f:
        f.write(cfgtxt)
