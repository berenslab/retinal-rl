def nourishment(name: str, states_definitions: str):
    return f"""\
ACTOR {name} : CustomInventory {{
    +INVENTORY.ALWAYSPICKUP

    // Different states used for different appearances, randomly chosen at spawn
    States {{
        Pickup:
            TNT1 A 0 ACS_NamedExecuteAlways("func_{name}")
            Stop
        {states_definitions}
        }}
}}"""


def poison(name: str, states_definitions: str):
    return nourishment(name, states_definitions)


def obstacle(name: str, states_definitions: str, radius: int = 24):
    return f"""\
ACTOR {name} : TorchTree {{
    Radius {radius}

    // Different states used for different appearances, randomly chosen at spawn
    States {{
        {states_definitions}\
    }}
}}"""


def distractor(name: str, states_definitions: str):
    return f"""\
ACTOR {name} : CustomInventory {{
    +INVENTORY.ALWAYSPICKUP

    // Different states used for different appearances, randomly chosen at spawn
    States {{
        {states_definitions}\
    }}
}}
"""


def states_template(index: int, texture_code: str):
    return f"Texture{index}: {texture_code} A -1\n\t"


def include(actor_name: str):
    return f'#include "actors/{actor_name}.dec"\n'
