def nourishment(name, states_definitions):
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


def poison(name, states_definitions):
    return nourishment(name, states_definitions)


def obstacle(name, states_definitions, radius=24):
    return """\
ACTOR {name} : TorchTree {{
    Radius {radius}
    
    // Different states used for different appearances, randomly chosen at spawn
    States {{
        {states_definitions}
    }}
}}""".format(
        name, radius, states_definitions
    )


def distractor(name, states_definitions):
    return """\
ACTOR {name} : CustomInventory {{
    +INVENTORY.ALWAYSPICKUP
    
    // Different states used for different appearances, randomly chosen at spawn
    States {{
        {states_definitions}
    }}
}}
""".format(
        name=name, states_definitions=states_definitions
    )


def include(actor_name):
    return '#include "actors/{0}.dec"\n"'.format(actor_name)
