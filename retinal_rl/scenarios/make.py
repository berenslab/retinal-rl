import os
from glob import glob
import subprocess
import os.path as osp

import yaml
import omg


### Load Config ###


def load_config(flnm):
    with open("scenarios/{0}.yaml".format(flnm)) as stream:
        return yaml.safe_load(stream)

object_types = ['nourishment','poison','obstacle','distractor']

### Building ACS files ###

def make_acs(cfg,actor_names,actor_num_textures):
    
    acs = "actor"

    # Directives
    acs +="""// Directives
#import "gathering.acs"
#include "zcommon.acs"

script "Load Config Information" OPEN {{

    // Metabolic variables
    metabolic_delay = {0};
    metabolic_damage = {1};
    """.format(cfg['metabolic']['delay'],cfg['metabolic']['damage'])

    for typ in object_types:
        xcfg = cfg[typ]
        acs += """
    // {0} variables
    {0}_unique = {1};
    {0}_init = {2};
    {0}_delay = {3};
    """.format(typ,len(xcfg['actors']),xcfg['init'],xcfg['delay'])

    acs += "\n// Loading arrays"

    for i,(actor_name,num_textures) in enumerate(zip(actor_names,actor_num_textures)):
        acs += """
    actor_names[{0}] = "{1}";
    actor_num_textures[{0}] = {2};
    """.format(i,actor_name,num_textures)

    acs += "\n}"

    return acs

### Building Decorate Files ###

decorate_pre = {}

decorate_pre['nourishment'] = """ACTOR {0} : CustomInventory
{{
    +INVENTORY.ALWAYSPICKUP
    States
    {{
    Pickup:
        TNT1 A 0 HealThing({1});
        Stop;
        """
decorate_pre['poison'] = """ACTOR {0} : CustomInventory
{{
    +INVENTORY.ALWAYSPICKUP
    States
    {{
    Pickup:
        TNT1 A 0 DamageThing({1});
        Stop;
        """
decorate_pre['obstacle'] = """ACTOR {0}
{{
    Radius 30
    Height 300
    +SOLID
    States
    {{
        """
decorate_pre['distractor'] = """ACTOR {0} : CustomInventory
{{
    +INVENTORY.ALWAYSPICKUP
    States
    {{
        """

def actor_code(i,j):
    # Convert k to caps alpha string
    return chr(65 + i) + texture_code(j)

def texture_code(j):
    # Convert k to a 3-digit alpha all-caps string
    return chr(65 + j // 26 ** 2) + chr(65 + (j // 26) % 26) + chr(65 + j % 26)

def decorate_actor(cfg,actor_name,typ,actor_idx,num_textures):

    cfg = cfg['typ']['actor_name']

    decorate = ""
    if typ == "nourishment":
        decorate += decorate_pre[typ].format(actor_name,cfg['healing'])
    elif type == "poison":
        decorate += decorate_pre[typ].format(actor_name,cfg['damage'])
    elif type == "obstacle":
        decorate += decorate_pre[typ].format(actor_name)
    elif type == "distractor":
        decorate += decorate_pre[typ].format(actor_name)
    else:
        raise ValueError("Invalid actor type: {0}".format(typ))

    # Multiline string for beginning of decorate file
    for j in range(num_textures):
        decorate += """
            Tex{0}:
                {0} A -1""".format(actor_code(actor_idx,j))

    decorate += """
            }
    }\n\n"""

    return decorate

### Loading Textures ###

def load_textures(wad,actor_idx,pngs):

    for j,png in enumerate(pngs):
        code = actor_code(actor_idx,j) + "A0"
        wad.sprites[code] = omg.Graphic(from_file=png)

### Creating Scenarios ###


def create_base_wad():

    bpth = "resources/base"
    wad = omg.WAD()

    grspth = osp.join(bpth,"grass.png")
    wndpth = osp.join(bpth,"wind.png")

    txtpth = osp.join(bpth,"TEXTMAP.txt")
    mappth = osp.join(bpth,"MAPINFO.txt")

    # Flats
    wad.ztextures['GRASS'] = omg.Graphic(from_file=grspth)

    # Data
    wad.data['WIND'] = omg.Lump(from_file=wndpth)
    wad.data['MAPINFO'] = omg.Lump(from_file=mappth)

    # Map preparation
    mpgrp = omg.LumpGroup()
    mpgrp['TEXTMAP'] = omg.Lump(from_file=txtpth)

    return wad,mpgrp

def make_scenario(scnnm):

    # Preloading
    cfg = load_config(scnnm)

    wad,mpgrp = create_base_wad()

    # Defining paths
    rscpth = "resources"
    bspth = osp.join(rscpth,"base")
    acspth = osp.join(bspth,"acs")

    bhipth = osp.join(rscpth,scnnm + ".acs")
    bhopth = osp.join(rscpth,scnnm + ".o")

    lbipth = osp.join(acspth,"retinal.acs")
    lbopth = osp.join(acspth,"retinal.o")

    # Building decorate and loading textures

    decorate = ""
    actor_names = []
    actor_num_textures = []

    actor_idx = 0
    for typ in object_types:
        for actor_name in cfg[typ]['actors']:

            pngpths = cfg[typ][actor_name]['textures']
            # get all pngs listend in pngpths and subdirs
            pngs = []
            for pngpth in pngpths:
                pngs += glob.glob(pngpth)

            num_textures = len(pngs)
            load_textures(wad,actor_idx,pngs)
            decorate += decorate_actor(cfg,actor_name,typ,actor_idx,num_textures)

            actor_idx += 1
            actor_names.append(actor_name)
            actor_num_textures.append(num_textures)

    # Decorate
    wad.data['DECORATE'] = omg.Lump(data=decorate.encode('utf-8'))

    # Write ACS
    with open(bhipth,'w') as f:
        f.write(make_acs(cfg,actor_names,actor_num_textures))

    # Compile ACS
    subprocess.call(["acc", "-i","/usr/share/acc", bhipth, bhopth])
    subprocess.call(["acc", "-i","/usr/share/acc", lbipth, lbopth])

    # Libraries
    wad.data[lbinm] = omg.Lump(from_file=lbipth)
    wad.libraries[lbonm] = omg.Lump(from_file=lbopth)

    # Map
    mpgrp['SCRIPTS'] = omg.Lump(from_file=bhipth)
    mpgrp['BEHAVIOR'] = omg.Lump(from_file=bhopth)
    wad.udmfmaps["MAP01"] = omg.UMapEditor(mpgrp).to_lumps()

    # Cleanup
    os.remove(bhopth)
    os.remove(lbopth)

    # Save to file
    wad.to_file(osp.join("scenarios",scnnm + ".wad"))
