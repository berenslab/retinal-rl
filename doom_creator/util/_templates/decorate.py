def nourishment(name:str, states_definitions:str):
    return """\
ACTOR {name} : CustomInventory {{
    +INVENTORY.ALWAYSPICKUP
    
    // Different states used for different appearances, randomly chosen at spawn
    States {{
        Pickup:
            TNT1 A 0 ACS_NamedExecuteAlways("func_{name}")
            Stop
        {states_definitions}
        }}
}}""".format(
        name=name, states_definitions=states_definitions
    )


def poison(name:str, states_definitions:str):
    return nourishment(name, states_definitions)


def obstacle(name:str, states_definitions:str, radius:int=24):
    return """\
ACTOR {name} : TorchTree {{
    Radius {radius}
    
    // Different states used for different appearances, randomly chosen at spawn
    States {{
        {states_definitions}\
    }}
}}""".format(
        name=name, radius=radius, states_definitions=states_definitions
    )


def distractor(name:str, states_definitions:str):
    return """\
ACTOR {name} : CustomInventory {{
    +INVENTORY.ALWAYSPICKUP
    
    // Different states used for different appearances, randomly chosen at spawn
    States {{
        {states_definitions}\
    }}
}}
""".format(
        name=name, states_definitions=states_definitions
    )

def states_template(index:int, texture_code:str):
    return "Texture{index}: {texture_code} A -1\n\t".format(
        index=index, texture_code=texture_code
    )

def include(actor_name:str):
    return '#include "actors/{0}.dec"\n'.format(actor_name)
