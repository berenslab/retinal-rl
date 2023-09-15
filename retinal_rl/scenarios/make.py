import os
import subprocess
import os.path as osp

import hiyapyco as hyaml
import omg
import shutil


### Load Config ###
def load_config(flnms):
    # list all config files
    flpths = []
    for flnm in flnms:
        flpths.append("resources/map_configs/{0}.yaml".format(flnm))

    # Load all yaml files listed in flnms and combine into a single dictionary, recursively combining keys
    cfg = hyaml.load(flpths,method=hyaml.METHOD_MERGE)
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

    for typ in object_types:
        tcfg = cfg['objects'][typ]
        object_variables_acs += object_variables_template.format(type=typ,unique=len(tcfg['actors']),init=tcfg['init'],delay=tcfg['delay'])


    actor_arrays_initialization = ""
    actor_arrays_template ="""
    actor_names[{index}] = "{actor_name}";
    actor_num_textures[{index}] = {num_textures};
    """
    for i,(actor_name,num_textures) in enumerate(zip(actor_names,actor_num_textures)):
        actor_arrays_initialization += actor_arrays_template.format(index=i,actor_name=actor_name,num_textures=num_textures)

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

def make_decorate(cfg,templates,actor_name,typ,actor_idx):

    acfg = cfg['actors'][actor_name]
    tex_code=actor_code(actor_idx,0)

    if typ == "nourishment":
        decorate=templates[typ].format(name=actor_name,healing=acfg['healing'],texture_code=tex_code)
    elif typ == "poison":
        decorate=templates[typ].format(name=actor_name,damage=acfg['damage'],texture_code=tex_code)
    elif typ == "obstacle":
        decorate=templates[typ].format(name=actor_name, radius=24,texture_code=tex_code)
    elif typ == "distractor":
        decorate=templates[typ].format(name=actor_name,texture_code=tex_code)
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

### Creating Scenarios ###
def make_scenario(flnms,scnnm=None,clean=True):

    # Preloading
    cfg = load_config(flnms)
    ocfg = cfg['objects']
    object_types = ocfg.keys()

    if scnnm is None:
        scnnm = "-".join(flnms)

    # Base directories
    rscdr = "resources"
    scndr = "scenarios"

    # Inupt Directories
    ibsdr = osp.join(rscdr,"base")
    iacsdr = osp.join(ibsdr,"acs")
    itxtdr = osp.join(rscdr,"textures")

    # Output Directories
    blddr = osp.join(scndr,"build")
    oroot = osp.join(blddr,scnnm)
    oacsdr = osp.join(oroot,"acs")
    omapdr = osp.join(oroot,"maps")
    osptdr = osp.join(oroot,"sprites")
    oactdr = osp.join(oroot,"actors")
    otxtdr = osp.join(oroot,"textures")

    # Remove exiting root directory
    if osp.exists(oroot):
        shutil.rmtree(oroot)

    # Create Directories
    os.makedirs(scndr,exist_ok=True)
    os.makedirs(oroot,exist_ok=True)
    os.makedirs(oacsdr,exist_ok=True)
    os.makedirs(omapdr,exist_ok=True)
    os.makedirs(osptdr,exist_ok=True)
    os.makedirs(oactdr,exist_ok=True)
    os.makedirs(otxtdr,exist_ok=True)

    # Textures
    shutil.copy(osp.join(ibsdr,"grass.png"),osp.join(otxtdr,"GRASS.png"))
    shutil.copy(osp.join(ibsdr,"wind.png"), osp.join(otxtdr,"WIND.png"))

    # Copy Data to Root
    shutil.copy(osp.join(ibsdr,"MAPINFO.txt"), osp.join(oroot,"MAPINFO.txt"))

    # Building decorate and loading textures
    dec_templates=load_decorate_templates()
    actor_names = []
    actor_num_textures = []

    actor_idx = 0
    for typ in object_types:
        tcfg = ocfg[typ]
        for actor_name in tcfg['actors']:

            pngpths = tcfg['actors'][actor_name]['textures']
            # get all pngs listend in pngpths and subdirs
            pngs = []
            for pngpth in pngpths:
                fllpth = osp.join(itxtdr,pngpth)
                # if pngpth is a png, add it
                if pngpth.endswith(".png"):
                    pngs.append(fllpth)
                elif osp.isdir(fllpth):
                    # if pngpth is a directory, recursively add all pngs
                    for root, _, files in os.walk(fllpth):
                        for file in files:
                            if file.endswith(".png"):
                                pngs.append(osp.join(root,file))
            pngs = pngs
                

            num_textures = len(pngs)

            for j,png in enumerate(pngs):
                code = actor_code(actor_idx,j) + "A0"
                # Copy png to sprite pth
                shutil.copy(png,osp.join(osptdr,code + ".png"))

            decorate = make_decorate(tcfg,dec_templates, actor_name,typ,actor_idx)
            # write decorate to actor pth
            with open(osp.join(oactdr,actor_name + ".dec"),'w') as f:
                f.write(decorate)

            actor_idx += 1
            actor_names.append(actor_name)
            actor_num_textures.append(num_textures)

    # Write decorate to root
    decorate = make_decorate_include(actor_names)

    with open(osp.join(oroot,"DECORATE.txt"),'w') as f:
        f.write(decorate)

    ## ACS ##

    # Defining paths

    bhipth = osp.join(oroot,scnnm + ".acs")
    bhopth = osp.join(oroot,scnnm + ".o")

    lbipth = osp.join(iacsdr,"retinal.acs")
    lbopth = osp.join(oacsdr,"RETINAL")

    # Copy retinal to acs pth
    shutil.copy(lbipth,oacsdr)

    tmppth = osp.join(ibsdr,"TEXTMAP.txt")

    # Write ACS
    with open(bhipth,'w') as f:
        acs = make_acs(cfg,actor_names,object_types,actor_num_textures)
        f.write(acs)

    # Compile ACS
    subprocess.call(["acc", "-i","/usr/share/acc", bhipth, bhopth])
    subprocess.call(["acc", "-i","/usr/share/acc", lbipth, lbopth])

    # Map Wad

    wad = omg.WAD()
    mpgrp = omg.LumpGroup()

    mpgrp['TEXTMAP'] = omg.Lump(from_file=tmppth)
    mpgrp['SCRIPTS'] = omg.Lump(from_file=bhipth)
    mpgrp['BEHAVIOR'] = omg.Lump(from_file=bhopth)
    wad.udmfmaps["MAP01"] = omg.UMapEditor(mpgrp).to_lumps()

    # Cleanup
    os.remove(bhopth)

    # Save wad to map dir
    wad.to_file(osp.join(omapdr,"MAP01.wad"))

    # Copy vizdoom config
    cnfnm = scnnm + ".cfg"
    shutil.copy(osp.join(ibsdr,"vizdoom.cfg"),osp.join(scndr,cnfnm))
    # add doom_scenario_path to beginning of cfg
    with open(osp.join(scndr,cnfnm),'r') as f:
        cfgtxt = f.read()
    cfgtxt = """# Settings copied from resources/base/vizdoom.cfg
doom_scenario_path = {0}.zip

""".format(scnnm) + cfgtxt
    with open(osp.join(scndr,cnfnm),'w') as f:
        f.write(cfgtxt)

    # zip build and save to scenarios
    shutil.make_archive(osp.join(scndr,scnnm),'zip',oroot)

    # If clean flag is set, remove build directory
    if clean:
        shutil.rmtree(oroot)
        # Also if the build directory is empty, remove it too
        if len(os.listdir(blddr)) == 0:
            shutil.rmtree(blddr)

