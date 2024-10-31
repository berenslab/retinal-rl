import os
import os.path as osp
import shutil
import subprocess
from typing import Dict, List, Optional
from zipfile import ZipFile, ZipInfo

import omg
from tqdm import tqdm

from doom_creator.util import config, templates
from doom_creator.util.directories import Directories


### Creating Scenarios ###
def make_scenario(
    cfg: config.Config,
    directories: Directories,
    scenario_name: Optional[str] = None,
):
    # Create Zip for output
    out_file = osp.join(directories.SCENARIO_OUT_DIR, scenario_name) + ".zip"
    if osp.exists(out_file):
        os.remove(out_file)
    s_zip = ZipFile(out_file, "x")

    # Create directories in zip
    dirs = ["acs", "maps", "sprites", "actors", "textures"]
    for directory_name in dirs:
        if hasattr(s_zip, "mkdir"):  # .mkdir was introduced in python3.11
            s_zip.mkdir(directory_name)
        else:
            zip_info = ZipInfo(directory_name + "/")
            zip_info.external_attr = 0o40775 << 16  # drwxrwxr-x
            s_zip.writestr(zip_info, "")

    # Textures
    s_zip.write(
        osp.join(directories.ASSETS_DIR, "grass.png"), osp.join("textures", "GRASS.png")
    )
    s_zip.write(
        osp.join(directories.ASSETS_DIR, "wind.png"), osp.join("textures", "WIND.png")
    )

    # Copy Data to Root
    s_zip.write(osp.join(directories.ASSETS_DIR, "MAPINFO.txt"), "MAPINFO.txt")

    # Building decorate and loading textures
    actor_names = []
    actor_num_textures = []

    include_decorate = ""
    actor_idx = 0
    for typ, type_cfg in tqdm(cfg.objects.items(), desc="Creating Objects"):
        for actor_name, actor_cfg in tqdm(
            type_cfg.actors.items(), desc="Creating " + typ.value, leave=False
        ):
            actor_name = actor_name.replace("-", "_")  # Make name ACS compatible
            # get all pngs listend in pngpths and subdirs
            png_pths = actor_cfg.textures
            pngs = get_pngs(osp.join(directories.CACHE_DIR, "textures"), png_pths)
            num_textures = len(pngs)

            sprite_names = [actor_code(actor_idx, i) for i in range(num_textures)]
            # Add pngs as sprites
            for j, png in tqdm(
                enumerate(pngs),
                desc="add textures for " + actor_name,
                leave=False,
                total=num_textures,
            ):
                s_zip.write(png, osp.join("sprites", sprite_names[j] + "A0.png"))

            actor_idx += 1
            actor_names.append(actor_name)
            actor_num_textures.append(num_textures)

            dec = make_actor_decorate(actor_name, typ, sprite_names)
            s_zip.writestr(osp.join("actors", actor_name + ".dec"), dec)
            include_decorate += templates.decorate.include(actor_name)

    # Write decorate include to root
    s_zip.writestr("DECORATE.txt", include_decorate)

    ## Create ACS ##

    # Defining pths
    if osp.exists(directories.BUILD_DIR):
        shutil.rmtree(directories.BUILD_DIR)
    os.mkdir(directories.BUILD_DIR)

    retinal_acs_pth = osp.join(directories.ASSETS_DIR, "acs", "retinal.acs")
    map_acs_pth = osp.join(directories.BUILD_DIR, scenario_name) + ".acs"
    retinal_comp_pth = osp.join(directories.BUILD_DIR, "retinal.o")
    map_comp_pth = map_acs_pth[:-3] + "o"  # Replace ".acs" ending with ".o"

    acs = make_acs(
        cfg.objects,
        actor_names,
        actor_num_textures,
        cfg.metabolic.delay,
        cfg.metabolic.damage,
        spawn_relative=cfg.spawn_objects.relative,
        spawn_range=cfg.spawn_objects.range,
    )
    with open(map_acs_pth, "w") as f:
        f.write(acs)

    # Compile ACS
    subprocess.call(["acc", "-i", "/usr/share/acc", retinal_acs_pth, retinal_comp_pth])
    subprocess.call(
        ["acc", "-i", "/usr/share/acc", "-i", directories.ASSETS_DIR, map_acs_pth]
    )

    # For completeness, add retinal and map acs to zip
    s_zip.write(retinal_comp_pth, osp.join("acs", "retinal.o"))
    s_zip.write(retinal_acs_pth, osp.join("acs", "retinal.acs"))
    s_zip.write(map_acs_pth, "behavior.acs")

    # Map Wad
    wad = omg.WAD()
    map_lump = omg.LumpGroup()
    map_lump["TEXTMAP"] = omg.Lump(
        from_file=osp.join(directories.ASSETS_DIR, "TEXTMAP.txt")
    )
    map_lump["BEHAVIOR"] = omg.Lump(from_file=map_comp_pth)
    wad.udmfmaps["MAP01"] = omg.UMapEditor(map_lump).to_lumps()

    # Save wad to map and add to zip
    map_pth = osp.join(directories.BUILD_DIR, "MAP01.wad")
    wad.to_file(map_pth)
    s_zip.write(map_pth, osp.join("maps", "MAP01.wad"))

    # Cleanup
    shutil.rmtree(directories.BUILD_DIR)

    # Copy vizdoom config
    config_name = scenario_name + ".cfg"
    # add doom_scenario_pth to beginning of cfg
    with open(osp.join(directories.SCENARIO_OUT_DIR, config_name), "w") as f:
        f.write(templates.vizdoom.config(scenario_name=scenario_name))


### Building ACS files ###
def make_acs(
    objects_cfg: Dict[config.ObjectType, config.ObjectTypeVars],
    actor_names: List[str],
    num_textures: int,
    metabolic_delay: int,
    metabolic_damage: int,
    spawn_relative: bool = False,
    spawn_range: float = 1000.0,
):
    """Creates the acs script determining spawning and behaviour of all actors"""
    object_variables_acs = ""
    actor_functions = ""

    for typ, type_cfg in objects_cfg.items():
        object_variables_acs += templates.acs.object_variables(
            typ=typ.value,
            unique=len(type_cfg.actors),
            init=type_cfg.init,
            delay=type_cfg.delay,
        )

        for actor_name, actor_cfg in type_cfg.actors.items():
            if typ is config.ObjectType.nourishment:
                actor_functions += templates.acs.heal_function(
                    actor_name,
                    values=values_list(actor_cfg.healing),
                )
            elif typ is config.ObjectType.poison:
                actor_functions += templates.acs.damage_function(
                    actor_name,
                    values=values_list(actor_cfg.damage),
                )

    actor_arrays_initialization = ""
    for i, (actor_name, num_textures) in enumerate(zip(actor_names, num_textures)):
        actor_arrays_initialization += templates.acs.actor_arrays(
            index=i, actor_name=actor_name, num_textures=num_textures
        )

    acs = templates.acs.general(
        metabolic_delay=metabolic_delay,
        metabolic_damage=metabolic_damage,
        object_variables=object_variables_acs,
        array_variables=actor_arrays_initialization,
        actor_functions=actor_functions,
        spawn_relative=spawn_relative,
        spawn_range=spawn_range,
    )

    return acs


def values_list(cfg_values):
    _values = cfg_values
    if isinstance(_values, int):
        _values = [_values]  # Ensure iterable
    return _values


### Building Decorate Files ###


def actor_code(i, j):
    """Creates a 4-digit alpha all-caps string, F is skipped in first digit"""
    if i >= 5:
        return chr(65 + 1 + i) + texture_code(j)
    return chr(65 + i) + texture_code(j)


def texture_code(j):
    """Convert j to a 3-digit alpha all-caps string"""
    return chr(65 + j // 26**2) + chr(65 + (j // 26) % 26) + chr(65 + j % 26)


def make_actor_decorate(
    actor_name: str, typ: config.ObjectType, sprite_names: list[str]
):
    """Returns the decorate description for an actor as a str.

    Keyword arguments:
    actor_name -- name of the actor
    typ -- (nourishment, poison, obstacle, distractor)
    Each sprite img is used as an individual state.
    """
    states = ""

    for i, sprite_name in enumerate(sprite_names):
        states += templates.decorate.states_template(index=i, texture_code=sprite_name)

    if typ is config.ObjectType.nourishment:
        decorate = templates.decorate.nourishment(
            name=actor_name, states_definitions=states
        )
    elif typ is config.ObjectType.poison:
        decorate = templates.decorate.poison(name=actor_name, states_definitions=states)
    elif typ is config.ObjectType.distractor:
        decorate = templates.decorate.distractor(
            name=actor_name, states_definitions=states
        )
    elif typ is config.ObjectType.obstacle:
        decorate = templates.decorate.obstacle(
            name=actor_name, states_definitions=states
        )
    else:
        raise ValueError(f"Invalid actor type: {typ}")

    return decorate


def get_pngs(base_pth: str, png_pths: list[str]):
    """Returns all .png files in subdirs of each png_pth (with base_pth used as root dir)"""
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
