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


def predator(name: str, states_definitions: str, speed: int):
    return f"""\
ACTOR {name} : Actor {{
    Speed {speed}
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


def random_move_predator_states_template(index: int, texture_code: str, actor_name: str):
    return f"""Texture{index}:
    {texture_code} A 1 A_Wander
    Goto Wander{index}
  Wander{index}:
    {texture_code} A 1 A_Wander
    {texture_code} A 1 A_LookEx(0, 0, 600, 0, 360, "See{index}")
    Loop
  See{index}:
    {texture_code} A 1 A_Chase("Melee{index}", "Wander{index}")
    Loop
  Melee{index}:
    {texture_code} A 0 ACS_NamedExecuteAlways("func_{actor_name}")
    Goto Wander{index}
"""

def include(actor_name: str):
    return f'#include "actors/{actor_name}.dec"\n'
